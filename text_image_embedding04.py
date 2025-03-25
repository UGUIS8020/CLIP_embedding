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
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any

# 本文とfigテキスト、fig画像全てを一回でembeddingする。
# summeryを生成する。
# 01,02の進化版

# 環境変数をロード
load_dotenv()

@dataclass
class Metadata:
    title: str = ""      # 元のファイル名（.txt付き）
    base_name: str = ""  # .txtを除いたベース名を追加
    item: str = ""
    content: str = ""
    summary: str = ""
    keywords: str = ""
    figure_contexts: Dict[str, str] = None
    grouped_figure_contexts: Dict[str, Dict] = None
    topic: str = ""
    figures: Dict[str, Figure] = field(default_factory=dict)

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
        self.topic_pattern = re.compile(r'topic\[(.*?)\]')
        self.figure_pattern = re.compile(r'Fig(\d+)')

    def process_file(self, file_path: str) -> Metadata:
        """テキストファイルを処理し、メタデータとエントリーを抽出"""
        metadata = Metadata()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # ファイル名からメタデータを抽出
            filename = os.path.basename(file_path)
            metadata.title = filename
            metadata.base_name = os.path.splitext(filename)[0]
            
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
        # Pineconeの初期化
        print("Pineconeの初期化中...")
        pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY")
        )
        
        # OpenAI APIの初期化
        print("OpenAI APIの初期化中...")
        openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 入力ディレクトリの設定
        input_directory = r"C:\Users\shibu\Desktop\website\CLIP_embedding\data"
        print(f"入力ディレクトリ: {input_directory}")
        processor = EmbeddingProcessor(client=openai_client)
        
        print("ファイルの処理を開始します...")
        processed_files = process_files(input_directory)
        print(f"処理されたファイル数: {len(processed_files)}")
        
        if not processed_files:
            print("処理するファイルがありません")
            exit()
        
        if input("embedding生成を開始しますか？ (y/n): ").lower() == 'y':
            all_vectors = []
            for metadata in processed_files:
                print(f"\n{metadata.title}の処理を開始...")
                vectors = create_vectors(metadata, processor)
                all_vectors.extend(vectors)
                print(f"  {metadata.title} の処理完了: {len(vectors)}個のベクトル生成")
                
            print(f"\n生成されたベクトル総数: {len(all_vectors)}")
            print("Pineconeへのアップロードを開始します...")
            
            upsert_to_pinecone(all_vectors)
            print("すべてのデータのアップロードが完了しました")
        else:
            print("処理を中止しました")
            
    except Exception as e:
        print("\nエラーが発生しました:")
        print(f"エラータイプ: {type(e).__name__}")
        print(f"エラーメッセージ: {str(e)}")
        import traceback
        print(f"スタックトレース:\n{traceback.format_exc()}")

def build_metadata_text(metadata_info: Metadata, fig_base_id: str) -> str:
    """メタデータテキストを構築"""
    metadata_text = ""
    if metadata_info.title:
        metadata_text += f"title: {metadata_info.title}\n"
    if metadata_info.item:
        metadata_text += f"item: {metadata_info.item}\n"
    if metadata_info.summary:
        metadata_text += f"summary: {metadata_info.summary}\n"
    return metadata_text

def build_metadata_fields(data: Entry, metadata_info: Metadata, fig_base_id: str) -> Dict:
    """メタデータフィールドを構築"""
    metadata_fields = {
        "category": "dental",
        "data_type": "text",
        "text": data.text,
        "summary": metadata_info.summary,
        "entry_type": data.type,
        "related_image_id": data.image_id
    }
    
    if metadata_info.item:
        metadata_fields["item"] = metadata_info.item
    if metadata_info.keywords:  # キーワードを追加
        metadata_fields["keywords"] = metadata_info.keywords    
    
    metadata_fields["related_content_id"] = f"{data.text_id}"
    
    
    return metadata_fields

def build_image_metadata_fields(data: Entry, metadata_info: Metadata, fig_base_id: str) -> Dict:
    """画像のメタデータフィールドを構築"""
    image_metadata_fields = {
        "category": "dental",
        "data_type": "image",
        "text": data.text,
        "summary": metadata_info.summary,
        "entry_type": data.type,
        "related_content_id": data.text_id
    }
    
    if metadata_info.item:
        image_metadata_fields["item"] = metadata_info.item
    if metadata_info.keywords:  # キーワードを追加
        image_metadata_fields["keywords"] = metadata_info.keywords
    
    image_metadata_fields["related_content_id"] = f"{data.text_id}"
    
    return image_metadata_fields

def create_vectors(metadata: Metadata, processor: EmbeddingProcessor) -> List[Dict]:
    vectors = []
    
    # メインテキストの処理（_contentを付ける）
    content_id = f"{metadata.base_name}_content"  # 例: Transplantation001_chapter02_001_content
    
    vectors.append({
        "id": content_id,  # 新しい本文ID形式
        "values": processor.get_text_embedding(metadata.content),
        "metadata": {
            "title": metadata.title,
            "type": "main_text",
            "text": metadata.content,
            "topic": metadata.topic,
            "is_parent": True
        }
    })

    # 図の処理
    for fig_id, figure in metadata.figures.items():
        image_id = f"{metadata.base_name}_{fig_id}_image"
        text_id = f"{metadata.base_name}_{fig_id}_text"
        
        # 画像のembedding
        if figure.image_path:
            vectors.append({
                "id": image_id,
                "values": processor.get_image_embedding(figure.image_path),
                "metadata": {
                    "title": metadata.title,
                    "type": "image",
                    "figure_id": fig_id,
                    "topic": metadata.topic,
                    "related_content_id": content_id,  # 新しい本文IDを参照
                    "is_parent": False
                }
            })
        
        # 図の説明文のembedding
        if figure.description:
            vectors.append({
                "id": text_id,
                "values": processor.get_text_embedding(figure.description),
                "metadata": {
                    "title": metadata.title,
                    "type": "figure_text",
                    "figure_id": fig_id,
                    "text": figure.description,
                    "topic": metadata.topic,
                    "related_content_id": content_id,  # 新しい本文IDを参照
                    "is_parent": False
                }
            })

if __name__ == "__main__":
    main()