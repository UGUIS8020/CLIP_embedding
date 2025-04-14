import os
import re
import glob
import numpy as np
import openai
import torch
import open_clip
from PIL import Image
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# 環境変数をロード
load_dotenv()

# ファイルの先頭付近に定数を定義
CATEGORY = "dental"
PINECONE_INDEX ="raiden" 

@dataclass
class Metadata:
    title: str = ""
    content: str = ""
    figure_descriptions: Dict[str, str] = field(default_factory=dict)
    topic: str = ""

@dataclass
class Entry:
    type: str
    number: str
    text: str
    text_id: str
    image_id: str
    metadata: Metadata
    base_name: str = ""

class TextProcessor:
    def __init__(self):
        # 正規表現パターンを修正して2文字のアルファベットにも対応        
        # self.figure_pattern = re.compile(r'\[(Fig\d+(?:[a-z]{1,2})?)\]') 
        # self.figure_pattern = re.compile(r'\[(Fig\d+(?:[a-z]+)?)\]')
        # self.figure_pattern = re.compile(r'\[(Fig\d+(?:[a-zA-Z]+)?)\]')
        self.figure_pattern = re.compile(r'\[(Fig\d+(?:[-a-zA-Z0-9_]+)?)\]')

    def process_file(self, file_path: str) -> List[Entry]:
        """テキストファイルを処理し、メタデータと図の説明を抽出"""
        entries = []
        try:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            metadata = Metadata()
            
            # ファイル名からメタデータを抽出
            filename = os.path.basename(file_path)
            metadata.title = filename.split('.')[0]
            
            # その後、title[]から抽出
            lines = text.split('\n')
            for line in lines:
                title_match = re.search(r'title\[(.*?)\]', line)
                if title_match:
                    metadata.title = title_match.group(1).strip()
            
            # 本文と図の説明を分離
            main_content, figure_content = self._split_content_and_figures(text)
            metadata.content = main_content.strip()
            
            # 図の説明を処理
            if figure_content:
                self._process_figure_descriptions(figure_content, metadata)
            
            # topicの抽出
            topic_match = re.search(r'topic\[(.*?)\]', text)
            if topic_match:
                metadata.topic = topic_match.group(1).strip()
            
            # Entryを作成する際にbase_nameを設定
            entry = Entry(
                type="content",
                number="",
                text=metadata.content,
                text_id=file_path,
                image_id="",
                metadata=metadata,
                base_name=base_name
            )
            entries.append(entry)
            
            # 図の説明文の処理
            for fig_id, description in metadata.figure_descriptions.items():
                if description:
                    entry = Entry(
                        type="figure_description",
                        number=fig_id,
                        text=description,
                        text_id=f"{base_name}_{fig_id}_desc",
                        image_id=f"{base_name}_{fig_id}_image",
                        metadata=metadata,
                        base_name=base_name
                    )
                    entries.append(entry)
            
        except Exception as e:
            print(f"テキスト抽出エラー: {e}")
        
        return entries

    def _split_content_and_figures(self, text: str) -> Tuple[str, str]:
        """本文と図の説明を分離"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if self.figure_pattern.match(line.strip()):
                return '\n'.join(lines[:i]).strip(), '\n'.join(lines[i:])
        return text.strip(), ""

    def _process_figure_descriptions(self, figure_content: str, metadata: Metadata):
        """図の説明文を処理"""
        current_figure = None
        current_text = []
        
        for line in figure_content.split('\n'):
            fig_match = self.figure_pattern.match(line.strip())
            if fig_match:
                if current_figure:
                    metadata.figure_descriptions[current_figure] = '\n'.join(current_text).strip()
                
                current_figure = fig_match.group(1)
                current_text = []
            elif line.strip() and current_figure:
                current_text.append(line.strip())
        
        if current_figure and current_text:
            metadata.figure_descriptions[current_figure] = '\n'.join(current_text).strip()

class ImageProcessor:
    def __init__(self, model, preprocess, device):
        self.model = model
        self.preprocess = preprocess
        self.device = device

    def get_embedding(self, image_path: str) -> Dict:
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
                "status": "success"
            }
        except Exception as e:
            print(f"画像エンベディングエラー: {e}")
            return {
                "vector": [0.0] * 1536,
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
        
        return (embedding_sum / count if count > 0 else embedding_sum).tolist()

def validate_environment():
    """環境変数の検証"""
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY")
    }
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise EnvironmentError(f"必要な環境変数が設定されていません: {', '.join(missing_vars)}")
    return required_vars

def initialize_services():
    """サービスの初期化"""
    env_vars = validate_environment()
    client = openai.OpenAI(api_key=env_vars["OPENAI_API_KEY"])
    pc = Pinecone(api_key=env_vars["PINECONE_API_KEY"])
    index = pc.Index(PINECONE_INDEX)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai")
    model.to(device)
    return client, index, model, preprocess, device

def create_vectors(data: Entry, related_ids: Dict[str, List[str]] = None) -> List[Dict]:
    vectors = []
    content_id = f"{data.base_name}_content"
    
    # タイプごとの重み設定
    type_weights = {
        "content": 1.0,      # メインテキスト（本文）は最大の重み
        "figure_description": 0.75,  # 画像説明は中程度の重み
        "image": 0.75         # 画像も中程度の重み
    }
    
    if data.type == "content":
        # メインコンテンツの場合
        related_images = []
        related_descriptions = []
        
        for fig_id in data.metadata.figure_descriptions.keys():
            if fig_id.startswith('Fig'):
                related_images.append(f"{data.base_name}_{fig_id}_image")
                related_descriptions.append(f"{data.base_name}_{fig_id}_desc")
        
        vectors.append({
            "id": content_id,
            "values": data.text_embedding,
            "metadata": {
                "type": data.type,
                "category": CATEGORY,  # カテゴリーを追加
                "title": data.metadata.title,
                "topic": data.metadata.topic,
                "text": data.text,
                "weight": type_weights["content"],
                "vector_id": content_id,
                "related_images": related_images,
                "related_descriptions": related_descriptions
            }
        })
    elif data.type == "figure_description":
        vectors.append({
            "id": data.text_id,
            "values": data.text_embedding,
            "metadata": {
                "type": data.type,
                "category": CATEGORY,  # カテゴリーを追加
                "text": data.text,
                "weight": type_weights["figure_description"],
                "vector_id": data.text_id,
                "related_content_id": content_id,
                "related_image_id": data.image_id,
                "title": data.metadata.title,
                "topic": data.metadata.topic
            }
        })
    elif data.type == "image":
        vectors.append({
            "id": data.image_id,
            "values": data.image_embedding,
            "metadata": {
                "type": data.type,
                "category": CATEGORY,   # カテゴリーを追加
                "weight": type_weights["image"],  # 重みを追加
                "vector_id": data.image_id,
                "related_content_id": content_id,
                "related_description_id": data.text_id,
                "title": data.metadata.title,
                "topic": data.metadata.topic
            }
        })
    
    return vectors

def main():
    try:
        # サービスの初期化
        client, index, model, preprocess, device = initialize_services()
        
        # プロセッサーの初期化
        text_processor = TextProcessor()
        image_processor = ImageProcessor(model, preprocess, device)
        text_embedding_processor = TextEmbeddingProcessor(client)
        
        # テキストファイルの処理
        txt_files = glob.glob("data/*.txt")
        if not txt_files:
            raise FileNotFoundError("data ディレクトリにtxtファイルが見つかりません")

        # 本文と画像説明の抽出
        all_entries = []
        for file_path in txt_files:
            entries = text_processor.process_file(file_path)
            all_entries.extend(entries)

        # 画像ファイルの確認
        image_files = glob.glob(os.path.join("data", "Fig*.jpg"))

        # 抽出結果の表示と確認
        print("\n=== 抽出結果の確認 ===")
        content_count = sum(1 for e in all_entries if e.type == "content")
        desc_count = sum(1 for e in all_entries if e.type == "figure_description")
        print(f"メインテキスト数: {content_count}")
        print(f"画像説明テキスト数: {desc_count}")
        print(f"画像ファイル数: {len(image_files)}")

        print("\n画像テキストと画像ファイルの対応:")
        print("\n画像テキスト:")
        for entry in all_entries:
            if entry.type == "figure_description":
                print(f"- {entry.text_id} -> {entry.image_id}")

        print("\n画像ファイル:")
        for img in sorted(image_files):
            print(f"- {os.path.basename(img)}")

        # 処理開始の確認
        confirm = input("\n上記のデータに対してembedding処理を開始しますか？ (y/n): ").lower().strip()
        if confirm != 'y':
            print("処理を中止します")
            return

        # embedding処理
        vectors_to_upsert = []
        
        # 各ファイルの処理
        for file_path in txt_files:
            print(f"\n処理中のファイル: {file_path}")
            
            # メインテキストと図の説明の処理
            entries = text_processor.process_file(file_path)
            for entry in entries:
                entry.text_embedding = text_embedding_processor.get_embedding(entry.text)
                vectors_to_upsert.extend(create_vectors(entry))
            
            # 画像の処理
            image_pattern = os.path.join("data", "Fig*.jpg")
            for img_path in glob.glob(image_pattern):
                img_name = os.path.basename(img_path)
                match = re.match(r'Fig(\d+)([a-z]*)\.jpg', img_name)
                if match:
                    fig_num = match.group(1)
                    fig_variant = match.group(2)
                    image_id = f"{entries[0].base_name}_Fig{fig_num}{fig_variant}_image"
                    
                    image_result = image_processor.get_embedding(img_path)
                    if image_result["status"] == "success":
                        image_entry = Entry(
                            type="image",
                            number=fig_num,
                            text="",
                            text_id=f"{entries[0].base_name}_Fig{fig_num}_desc",
                            image_id=image_id,
                            metadata=entries[0].metadata,
                            base_name=entries[0].base_name
                        )
                        image_entry.image_embedding = image_result["vector"]
                        vectors_to_upsert.extend(create_vectors(image_entry))
                        print(f"画像のembedding生成完了: {image_id}")
                    else:
                        print(f"画像の処理に失敗: {img_path}")
                
        # 処理結果の表示
        print("\n=== 処理結果 ===")
        print(f"処理したベクトル数: {len(vectors_to_upsert)}")
        print("\n各データの重みと内訳:")
        
        # 型別のカウントと重みの表示
        type_counts = {}
        for v in vectors_to_upsert:
            v_type = v["metadata"]["type"]
            weight = v["metadata"]["weight"]
            if v_type not in type_counts:
                type_counts[v_type] = {
                    "count": 0,
                    "weight": weight
                }
            type_counts[v_type]["count"] += 1
            
        for t, info in type_counts.items():
            print(f"- {t}:")
            print(f"  - 件数: {info['count']}")
            print(f"  - 重み: {info['weight']}")

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

if __name__ == "__main__":
    main()