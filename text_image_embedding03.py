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

@dataclass
class Metadata:
    title: str = ""
    content: str = ""
    figure_descriptions: Dict[str, str] = field(default_factory=dict)

class TextProcessor:
    def __init__(self):
        self.figure_pattern = re.compile(r'\[(Fig\d+[a-z]?)\]')

    def process_file(self, file_path: str) -> Metadata:
        """テキストファイルを処理し、メタデータと図の説明を抽出"""
        metadata = Metadata()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # ファイル名からメタデータを抽出
            metadata.title = os.path.splitext(os.path.basename(file_path))[0]
            
            # 本文と図の説明を分離
            main_content, figure_content = self._split_content_and_figures(text)
            metadata.content = main_content.strip()
            
            # 図の説明を処理
            if figure_content:
                self._process_figure_descriptions(figure_content, metadata)
            
            return metadata
        except Exception as e:
            print(f"テキスト抽出エラー: {e}")
            return metadata

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
    index = pc.Index("raiden02")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai")
    model.to(device)
    return client, index, model, preprocess, device

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
        
        vectors_to_upsert = []
        
        # 各ファイルの処理
        for file_path in txt_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            print(f"\n処理中のファイル: {file_path}")
            
            # メインテキストと図の説明の処理
            metadata = text_processor.process_file(file_path)
            if metadata.content:
                content_id = f"{base_name}_content"
                content_emb = text_embedding_processor.get_embedding(metadata.content)
                vectors_to_upsert.append({
                    "id": content_id,
                    "values": content_emb,
                    "metadata": {
                        "type": "content",
                        "title": metadata.title,
                        "text": metadata.content
                    }
                })
                print(f"メインテキストのembedding生成完了: {content_id}")
            
            # 図の説明文の処理
            for fig_id, description in metadata.figure_descriptions.items():
                if description:
                    fig_desc_id = f"{base_name}_{fig_id}_desc"
                    fig_desc_emb = text_embedding_processor.get_embedding(description)
                    vectors_to_upsert.append({
                        "id": fig_desc_id,
                        "values": fig_desc_emb,
                        "metadata": {
                            "type": "figure_description",
                            "title": metadata.title,
                            "figure_id": fig_id,
                            "text": description,
                            "related_content_id": content_id,
                            "related_image_id": f"{base_name}_{fig_id}_image"
                        }
                    })
                    print(f"図の説明文のembedding生成完了: {fig_desc_id}")
            
            # 画像の処理
            image_pattern = os.path.join("data", "Fig*.jpg")
            for img_path in glob.glob(image_pattern):
                img_name = os.path.basename(img_path)
                match = re.match(r'Fig(\d+)([a-z]*)\.jpg', img_name)
                if match:
                    fig_num = match.group(1)
                    fig_variant = match.group(2)
                    image_id = f"{base_name}_Fig{fig_num}{fig_variant}_image"
                    
                    image_result = image_processor.get_embedding(img_path)
                    if image_result["status"] == "success":
                        vectors_to_upsert.append({
                            "id": image_id,
                            "values": image_result["vector"],
                            "metadata": {
                                "type": "image",
                                "title": metadata.title,
                                "image_path": img_path,
                                "related_content_id": content_id,
                                "related_description_id": f"{base_name}_Fig{fig_num}_desc"
                            }
                        })
                        print(f"画像のembedding生成完了: {image_id}")
                    else:
                        print(f"画像の処理に失敗: {img_path}")
                
        # 処理結果の表示
        print("\n=== 処理結果 ===")
        print(f"処理したベクトル数: {len(vectors_to_upsert)}")
        print("内訳:")
        
        # 型別のカウント
        type_counts = {}
        for v in vectors_to_upsert:
            v_type = v["metadata"]["type"]
            type_counts[v_type] = type_counts.get(v_type, 0) + 1
            
        for t, count in type_counts.items():
            print(f"- {t}: {count}")

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