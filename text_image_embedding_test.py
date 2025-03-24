import os
import re
import glob
import numpy as np
import openai
import torch
import open_clip
import traceback
from PIL import Image
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple, Any

# 本文とfigテキスト、fig画像全てを一回でembeddingする。
# summeryを生成する。
# 01,02の進化版

# 環境変数をロード
load_dotenv()

@dataclass
class Metadata:
    title: str = ""
    item: str = ""
    content: str = ""
    content_summary: str = ""
    keywords: str = ""
    figure_contexts: Dict[str, str] = None
    grouped_figure_contexts: Dict[str, Dict] = None

    def __post_init__(self):
        self.figure_contexts = {}
        self.grouped_figure_contexts = {}
        
    def set_item(self, value: str):
        """itemを設定するメソッド"""
        self.item = value
        print(f"itemを設定: {value}")  # デバッグ用

@dataclass
class Entry:
    type: str  # "Fig" または "case"
    number: str
    text: str
    text_id: str
    image_id: str
    metadata: Metadata

class TextProcessor:
    def __init__(self):
        self.all_entries: Dict[str, Entry] = {}
        self.all_metadata: Dict[str, Metadata] = {}

    def process_file(self, file_path: str) -> Metadata:
        """テキストファイルを処理し、メタデータとエントリーを抽出"""
        metadata = Metadata()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # ファイル名からメタデータを抽出
            filename = os.path.basename(file_path)
            metadata.title = filename.split('.')[0]
            
            # title[]とitem[]の抽出
            lines = text.split('\n')
            for line in lines:
                title_match = re.search(r'title\[(.*?)\]', line)
                if title_match:
                    metadata.title = title_match.group(1).strip()
                
                item_match = re.search(r'item\[(.*?)\]', line)
                if item_match:
                    metadata.set_item(item_match.group(1).strip())
                    print(f"itemを抽出: {metadata.item}")  # デバッグ用
            
            # 説明文部分の開始を検出
            description_start = self._find_description_start(text)
            if description_start != -1:
                self._process_text_sections(text, description_start, metadata, filename)
            else:
                # 説明文がない場合は全体を本文として扱う
                metadata.content = text.strip()
                print(f"\n本文テキストの抽出:")
                print(f"抽出された本文:\n{text.strip()}\n")
            
            return metadata

        except Exception as e:
            print(f"テキスト抽出エラー: {e}")
            print(traceback.format_exc())
            return metadata

    def _find_description_start(self, text: str) -> int:
        """説明文部分の開始位置を検出"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if re.match(r'^\[(?:Fig|case)\d+[a-z]?\]', line.strip()):
                return i
        return -1

    def _process_text_sections(self, text: str, description_start: int, metadata: Metadata, filename: str):
        """テキストセクションを処理し、メタデータとエントリーを抽出"""
        lines = text.split('\n')
        
        # description_startが-1の場合は、全体を本文として扱う
        main_text_end = description_start if description_start != -1 else len(lines)
        
        # 本文テキストの抽出
        main_text = '\n'.join(lines[:main_text_end]).strip()
        print(f"\nデバッグ - 本文テキスト:\n{main_text}")  # デバッグ用
        
        if main_text:
            # item[...]の形式でアイテム情報を抽出
            item_match = re.search(r'item\[(.*?)\]', main_text)
            print(f"デバッグ - item_matchの結果: {item_match}")  # デバッグ用
            
            if item_match:
                item_value = item_match.group(1).strip()
                metadata.set_item(item_value)  # 新しいメソッドを使用
                print(f"\nアイテム情報の抽出:")
                print(f"抽出されたアイテム: {metadata.item}")
            else:
                print(f"警告: itemの形式が見つかりませんでした")  # デバッグ用
            
            metadata.content = main_text
            print(f"\n本文テキストの抽出:")
            print(f"抽出された本文:\n{main_text}\n")
        
        # 説明文部分の抽出と処理（description_startが有効な場合のみ）
        if description_start != -1:
            description_text = '\n'.join(lines[description_start:])
            self._process_description_sections(description_text, metadata, filename)

    def _process_description_sections(self, description_text: str, metadata: Metadata, filename: str):
        """説明文セクションを処理し、エントリーを生成"""
        sections = re.split(r'(\[Fig\d+[a-z]?\]|\[case\d+[a-z]?\])', description_text)
        current_ref = None
        current_text = []
        fig_count = 0
        case_count = 0
        
        for section in sections:
            if not section.strip():
                continue
                
            ref_match = re.match(r'\[(Fig|case)(\d+[a-z]?)\]', section)
            if ref_match:
                self._save_current_section(current_ref, current_text, metadata)
                
                ref_type = ref_match.group(1)
                ref_num = ref_match.group(2)
                current_ref = f"{ref_type}{ref_num}"
                current_text = []
                
                if ref_type == 'Fig':
                    fig_count += 1
                elif ref_type == 'case':
                    case_count += 1
            else:
                current_text.append(section)
        
        self._save_current_section(current_ref, current_text, metadata)
        
        # エントリーの生成
        self._create_entries(metadata, filename)
        
        print(f"\n{filename}: {fig_count}個の [FigX] と {case_count}個の [caseX] 説明文を取得")

    def _save_current_section(self, current_ref: str, current_text: List[str], metadata: Metadata):
        """現在のセクションを保存"""
        if current_ref and current_text:
            context = ''.join(current_text).strip()
            if context:
                metadata.figure_contexts[current_ref] = context
                print(f"\n{current_ref}のコンテキスト抽出:")
                print(f"抽出されたコンテキスト: {context[:100]}...")

    def _create_entries(self, metadata: Metadata, filename: str):
        """エントリーを生成して保存"""
        for ref_id, context in metadata.figure_contexts.items():
            ref_type = ref_id[:3] if ref_id.startswith('Fig') else ref_id[:4]
            number = ref_id[3:] if ref_id.startswith('Fig') else ref_id[4:]
            
            base_id = f"{filename}_{ref_id}"
            text_id = f"{base_id}_text"
            image_id = f"{base_id}_image"
            
            self.all_entries[base_id] = Entry(
                type=ref_type,
                number=number,
                text=context,
                text_id=text_id,
                image_id=image_id,
                metadata=metadata
            )

class ImageProcessor:
    def __init__(self, model, preprocess, device):
        self.model = model
        self.preprocess = preprocess
        self.device = device

    def get_embedding(self, image_path: str) -> Dict[str, Any]:
        """画像のembeddingを生成"""
        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_embedding = self.model.encode_image(image)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            vector = image_embedding.cpu().numpy().tolist()[0]
            expanded_vector = np.concatenate([vector, np.zeros(1536-len(vector))]).tolist()
            return {
                "vector": expanded_vector,
                "model": "open_clip",
                "status": "success"
            }
        except Exception as e:
            print(f"画像エンベディングエラー: {e}")
            return {
                "vector": [0.0] * 1536,
                "model": "open_clip",
                "status": "error",
                "error_message": str(e)
            }

class TextEmbeddingProcessor:
    def __init__(self, client):
        self.client = client
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )

    def get_embedding(self, text: str) -> List[float]:
        """テキストのembeddingを生成"""
        if isinstance(text, dict):
            text = ' '.join(str(v) for v in text.values())
        elif not isinstance(text, str):
            text = str(text)

        chunks = self.text_splitter.split_text(text)
        embedding_sum = np.zeros(1536)
        count = 0
        
        for chunk in chunks:
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk
                )
                embedding_sum += np.array(response.data[0].embedding)
                count += 1
            except Exception as e:
                print(f"Embedding生成エラー: {e}")
                print(traceback.format_exc())
        
        return (embedding_sum / count if count > 0 else embedding_sum).tolist()

class SummaryGenerator:
    def __init__(self, client):
        self.client = client

    def generate_summary(self, content: str, max_tokens: int = 500) -> str:
        """本文の要約を生成"""
        if not content or len(content.strip()) < 50:
            return ""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "あなたは歯科医学の専門家です。与えられた歯科医学のテキストを、150文字程度に要約してください"},
                    {"role": "user", "content": f"以下の歯科医学の文章を150字程度で要約してください。要約は完全な文章で終わるようにしてください：\n\n{content}"}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            summary = response.choices[0].message.content.strip()
            
            # 要約が途中で切れていないか確認
            if summary and not summary.endswith(('。', '．', '.',  '）', ')')):
                print("警告: 要約が途中で切れている可能性があります")
                # 最後の完全な文で終わるように調整
                last_sentence = summary.rfind('。')
                if last_sentence != -1:
                    summary = summary[:last_sentence + 1]
            
            return summary
        except Exception as e:
            print(f"要約生成エラー: {e}")
            return ""

    def generate_keywords(self, content: str) -> str:
        """本文からキーワードを生成"""
        if not content or len(content.strip()) < 50:
            return ""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                     {"role": "system", "content": """歯科医学の専門家として、以下の文章から重要なキーワードを抽出してください。
                            以下の点を特に重視してください：
                            - 術式の具体的な内容（例：大臼歯2歯の同時移植）
                            - 歯の本数、部位の情報
                            - 治療手順や手技の専門用語
                            - 重要な診断情報や治療結果
                            キーワードはカンマ区切りで返してください。"""},
                    {"role": "user", "content": content}
                ]
            )
            keywords = response.choices[0].message.content.strip()
            
            return keywords
        except Exception as e:
            print(f"キーワード生成エラー: {e}")
            return ""


def validate_environment():
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY")
    }
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise EnvironmentError(f"必要な環境変数が設定されていません: {', '.join(missing_vars)}")
    return required_vars

# 初期化処理をまとめる
def initialize_services():
    env_vars = validate_environment()
    client = openai.OpenAI(api_key=env_vars["OPENAI_API_KEY"])
    pc = Pinecone(api_key=env_vars["PINECONE_API_KEY"])
    index = pc.Index("raiden02")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    returned_values = open_clip.create_model_and_transforms(
        "ViT-B/32",
        pretrained="openai",
        jit=True,  # JITコンパイルを有効化
        force_quick_gelu=True  # QuickGELUを強制的に有効化
    )
    
    if len(returned_values) == 2:
        model, preprocess = returned_values
    elif len(returned_values) == 3:
        model, preprocess, _ = returned_values
    else:
        raise ValueError("open_clip.create_model_and_transforms からの予期しない戻り値の数です")
    
    model.to(device)
    return client, index, model, preprocess, device

def get_image_embedding(image_path, model, preprocess, device):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = model.encode_image(image)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        # 512次元を1536次元に拡張
        vector = image_embedding.cpu().numpy().tolist()[0]
        expanded_vector = np.concatenate([vector, np.zeros(1536-len(vector))]).tolist()
        return {
            "vector": expanded_vector,
            "model": "open_clip",
            "status": "success"
        }
    except Exception as e:
        print(f"画像エンベディングエラー: {e}")
        return {
            "vector": [0.0] * 1536,  # 1536次元のゼロベクトル
            "model": "open_clip",
            "status": "error",
            "error_message": str(e)
        }

# 512次元ベクトルを 1536次元に拡張（ゼロ埋め）
def expand_embedding(embedding_dict, target_dim=1536):
    vector = embedding_dict["vector"]
    extra_dims = target_dim - len(vector)
    expanded_vector = np.concatenate([vector, np.zeros(extra_dims)]).tolist()
    
    return {
        "vector": expanded_vector,
        "original_dim": len(vector),
        "model": embedding_dict["model"],
        "status": embedding_dict["status"]
    }

def get_text_embedding(text, client, chunk_size=1000, chunk_overlap=200):
    if isinstance(text, dict):
        text = ' '.join(str(v) for v in text.values())
    elif not isinstance(text, str):
        text = str(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    # 逐次的に平均を計算
    embedding_sum = np.zeros(1536)
    count = 0
    
    for chunk in chunks:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            embedding_sum += np.array(response.data[0].embedding)
            count += 1
        except Exception as e:
            print(f"Embedding生成エラー: {e}")
            print(traceback.format_exc())
    
    return (embedding_sum / count if count > 0 else embedding_sum).tolist()

def main():
    try:
        # サービスの初期化
        client, index, model, preprocess, device = initialize_services()
        
        # プロセッサーの初期化
        text_processor = TextProcessor()
        image_processor = ImageProcessor(model, preprocess, device)
        text_embedding_processor = TextEmbeddingProcessor(client)
        summary_generator = SummaryGenerator(client)
        
        # テキストファイルの処理
        txt_files = glob.glob("data/*.txt")
        if not txt_files:
            raise FileNotFoundError("data ディレクトリにtxtファイルが見つかりません")
        
        # 各ファイルの処理
        for file_path in txt_files:
            # メタデータの抽出（図表処理用）
            metadata = text_processor.process_file(file_path)
            text_processor.all_metadata[file_path] = metadata
            
            # カウントの表示
            fig_count = sum(1 for k, v in text_processor.all_entries.items() if v.type == "Fig")
            case_count = sum(1 for k, v in text_processor.all_entries.items() if v.type == "case")
            print(f"{file_path}: {fig_count}個の [FigX] と {case_count}個の [caseX] 説明文を取得")
            
            # メタデータの表示
            if metadata.title or metadata.item:
                print(f"  メタデータ: title={metadata.title}, item={metadata.item}")

        # テキストと画像の詳細を収集
        text_entries = {"Fig": {}, "case": {}}
        image_entries = {"Fig": {}, "case": {}}
        missing_images = []
        entries_to_process = []

        # エントリーの処理
        for base_id, data in text_processor.all_entries.items():
            entry_type = data.type
            number = data.number
            
            # 数字部分と変異部分を分離
            match = re.search(r'(\d+)([a-z]?)', number)
            if match:
                base_num = match.group(1)
                variation = match.group(2)
                
                # テキストエントリーを登録
                if base_num not in text_entries[entry_type]:
                    text_entries[entry_type][base_num] = set()
                if variation:
                    text_entries[entry_type][base_num].add(variation)
                elif not text_entries[entry_type][base_num]:
                    text_entries[entry_type][base_num].add('')
            
                # 画像の存在確認
                text_file_part = base_id.split(f'_{entry_type}')[0]
                image_path = f"data/{entry_type}{base_num}{variation}.jpg"
                
                if os.path.exists(image_path):
                    # 画像が存在する場合
                    if base_num not in image_entries[entry_type]:
                        image_entries[entry_type][base_num] = set()
                    if variation:
                        image_entries[entry_type][base_num].add(variation)
                    elif not image_entries[entry_type][base_num]:
                        image_entries[entry_type][base_num].add('')
                    
                    # embeddingを後で計算するためのデータを保存
                    entries_to_process.append({
                        "base_id": base_id,
                        "text_data": data,
                        "image_path": image_path,
                        "text_file_part": text_file_part,
                        "entry_type": entry_type,
                        "base_num": base_num,
                        "variation": variation
                    })
                else:
                    missing_images.append(image_path)
            
            print(f"- {base_id}の確認完了")

        # 統計情報の表示
        total_fig_text = sum(1 for _, v in text_processor.all_entries.items() if v.type == "Fig")
        total_case_text = sum(1 for _, v in text_processor.all_entries.items() if v.type == "case")
        total_fig_image = sum(1 for item in entries_to_process if item["entry_type"] == "Fig")
        total_case_image = sum(1 for item in entries_to_process if item["entry_type"] == "case")
        
        print(f"\n確認したテキスト数: Fig={total_fig_text}, case={total_case_text}")
        print(f"確認した画像数: Fig={total_fig_image}, case={total_case_image}")
        
        # Fig と case それぞれの詳細を表示
        for entry_type in ["Fig", "case"]:
            print(f"\n=== {entry_type}の詳細 ===")
            all_base_numbers = sorted(set(text_entries[entry_type].keys()) | set(image_entries[entry_type].keys()))
            
            print(f"\nテキストの{entry_type}:")
            for base_num in all_base_numbers:
                variations = text_entries[entry_type].get(base_num, set())
                if variations:
                    var_str = ', '.join([f'{entry_type}{base_num}{v}' if v else f'{entry_type}{base_num}' for v in sorted(variations)])
                    print(f"{entry_type}{base_num}: {var_str}")
                else:
                    print(f"{entry_type}{base_num}: なし")

            print(f"\n画像の{entry_type}:")
            for base_num in all_base_numbers:
                variations = image_entries[entry_type].get(base_num, set())
                if variations:
                    var_str = ', '.join([f'{entry_type}{base_num}{v}' if v else f'{entry_type}{base_num}' for v in sorted(variations)])
                    print(f"{entry_type}{base_num}: {var_str}")
                else:
                    print(f"{entry_type}{base_num}: なし")

            # 不一致の確認
            has_mismatch = False
            print(f"\n=== {entry_type}の不一致確認 ===")
            for base_num in all_base_numbers:
                text_vars = text_entries[entry_type].get(base_num, set())
                image_vars = image_entries[entry_type].get(base_num, set())
                if text_vars != image_vars:
                    has_mismatch = True
                    print(f"\n{entry_type}{base_num}の不一致:")
                    if text_vars - image_vars:
                        print(f"  テキストにあって画像にない: {', '.join([f'{entry_type}{base_num}{v}' if v else f'{entry_type}{base_num}' for v in sorted(text_vars - image_vars)])}")
                    if image_vars - text_vars:
                        print(f"  画像にあってテキストにない: {', '.join([f'{entry_type}{base_num}{v}' if v else f'{entry_type}{base_num}' for v in sorted(image_vars - text_vars)])}")

            if not has_mismatch:
                print(f"\nすべての{entry_type}とそのバリエーションが一致しています。")

        if missing_images:
            print("\n見つからなかった画像ファイル:")
            for path in missing_images:
                print(f"- {path}")

        # ユーザー確認 - embeddingを行うかどうか
        proceed = input("\nembedding計算を開始しPineconeにデータを送信しますか？ (y/n): ").lower()
        if proceed != 'y':
            print("処理を中止します")
            return

        # 要約とキーワードの生成
        print("\n本文の要約とキーワードを生成中...")
        for file_path, metadata in text_processor.all_metadata.items():
            if metadata.content:
                # 要約の生成
                metadata.content_summary = summary_generator.generate_summary(metadata.content)
                print(f"  {file_path}の本文要約を生成しました ({len(metadata.content_summary)}文字)")
                
                # キーワードの生成を追加
                metadata.keywords = summary_generator.generate_keywords(metadata.content)
                print(f"  {file_path}のキーワードを生成しました")

        # embedding処理のための変数を初期化
        vectors_to_upsert = []
        text_count = 0
        image_count = 0
        content_count = 0

        # 本文のembedding生成
        print("\n本文のエンベディングを生成中...")
        for file_path, metadata in text_processor.all_metadata.items():
            if metadata.content:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                content_id = f"{base_name}_content"
                
                content_emb = text_embedding_processor.get_embedding(metadata.content)
                content_count += 1
                
                content_metadata = {
                    "category": "dental",
                    "data_type": "content",
                    "item": metadata.item,
                    "content_summary": metadata.content_summary,
                    "content_text": metadata.content,
                    "keywords": metadata.keywords  # キーワードをメタデータに追加
                }
                
                related_figs = [k for k, v in text_processor.all_entries.items() if k.startswith(base_name)]
                if related_figs:
                    content_metadata["related_figures"] = related_figs
                
                vectors_to_upsert.append((
                    content_id,
                    content_emb,
                    content_metadata
                ))
                
                print(f"  {content_id}の本文エンベディングを生成しました")

        # 図表のembedding計算
        print("\n図表のembedding計算を開始します...")
        for item in entries_to_process:
            base_id = item["base_id"]
            data = item["text_data"]
            image_path = item["image_path"]
            text_file_part = item["text_file_part"]
            entry_type = item["entry_type"]
            base_num = item["base_num"]
            variation = item["variation"]
            
            # メタデータ情報の構築
            metadata_info = data.metadata
            fig_base_id = f"{entry_type}{base_num}"
            
            # メタデータテキストの構築
            metadata_text = build_metadata_text(metadata_info, fig_base_id)
            
            # テキストのembedding取得
            print(f"テキストのembedding計算中: {data.text_id}")
            combined_text = metadata_text + data.text if metadata_text else data.text
            text_emb = text_embedding_processor.get_embedding(combined_text)
            text_count += 1
            
            # メタデータフィールドの構築
            metadata_fields = build_metadata_fields(data, metadata_info, fig_base_id)
            
            vectors_to_upsert.append((
                data.text_id,
                text_emb,
                metadata_fields
            ))
            
            # 画像のembedding取得
            print(f"画像のembedding計算中: {image_path}")
            image_result = image_processor.get_embedding(image_path)
            if image_result["status"] == "success":
                image_count += 1
                image_emb = image_result["vector"]
                image_id = f"{text_file_part}_{entry_type}{base_num}{variation}_image"
                
                # 画像のメタデータフィールドの構築
                image_metadata_fields = build_image_metadata_fields(data, metadata_info, fig_base_id)
                
                vectors_to_upsert.append((
                    image_id,
                    image_emb,
                    image_metadata_fields
                ))
            
            print(f"- {base_id}の処理完了")

        print(f"\n処理したテキスト数: {text_count}")
        print(f"処理した画像数: {image_count}")
        print(f"処理した本文数: {content_count}")

        # Pineconeへのアップロード
        if vectors_to_upsert:
            try:
                index.upsert(vectors=vectors_to_upsert)
                print(f"\n{len(vectors_to_upsert)}個のベクトルをPineconeに保存しました")
            except Exception as e:
                print(f"\nPineconeへの保存中にエラーが発生: {e}")
        else:
            print("\nアップロードするデータがありません")

    except Exception as e:
        print(f"\n処理中にエラーが発生: {e}")
        print(traceback.format_exc())

def build_metadata_text(metadata_info: Metadata, fig_base_id: str) -> str:
    """メタデータテキストを構築"""
    metadata_text = ""
    if metadata_info.title:
        metadata_text += f"title: {metadata_info.title}\n"
    if metadata_info.item:
        metadata_text += f"item: {metadata_info.item}\n"
    if metadata_info.content_summary:
        metadata_text += f"content_summary: {metadata_info.content_summary}\n"
    return metadata_text

def build_metadata_fields(data: Entry, metadata_info: Metadata, fig_base_id: str) -> Dict:
    """メタデータフィールドを構築"""
    metadata_fields = {
        "category": "dental",
        "data_type": "text",
        "text": data.text,
        "entry_type": data.type,
        "related_image_id": data.image_id
    }
    
    if metadata_info.item:
        metadata_fields["item"] = metadata_info.item
    if metadata_info.content_summary:
        metadata_fields["content_summary"] = metadata_info.content_summary
    if metadata_info.keywords:  # キーワードを追加
        metadata_fields["keywords"] = metadata_info.keywords
    
    content_id = f"{data.text_id}_content"
    metadata_fields["related_content_id"] = content_id
    
    return metadata_fields

def build_image_metadata_fields(data: Entry, metadata_info: Metadata, fig_base_id: str) -> Dict:
    """画像のメタデータフィールドを構築"""
    image_metadata_fields = {
        "category": "dental",
        "data_type": "image",
        "text": data.text,
        "entry_type": data.type,
        "related_text_id": data.text_id
    }
    
    if metadata_info.item:
        image_metadata_fields["item"] = metadata_info.item
    if metadata_info.content_summary:
        image_metadata_fields["content_summary"] = metadata_info.content_summary
    
    content_id = f"{data.text_id}_content"
    image_metadata_fields["related_content_id"] = content_id
    
    return image_metadata_fields

if __name__ == "__main__":
    main()