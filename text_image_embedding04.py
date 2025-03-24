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

# 環境変数をロード
load_dotenv()

@dataclass
class FigureData:
    figure_id: str
    image_path: str = ""
    description: str = ""
    references: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Metadata:
    title: str = ""
    topic: str = ""
    content: str = ""
    content_summary: str = ""
    figures: Dict[str, FigureData] = field(default_factory=dict)

    def add_figure(self, figure: FigureData):
        self.figures[figure.figure_id] = figure

class TextProcessor:
    def __init__(self):
        self.figure_pattern = re.compile(r'^\[(?:Fig|case)\d+[a-z]?\]')
        self.reference_pattern = re.compile(r'\(Fig\d+[a-z]?\)')

class ImageProcessor:
    def __init__(self, model, preprocess, device):
        self.model = model
        self.preprocess = preprocess
        self.device = device

    def get_embedding(self, image_path: str) -> List[float]:
        """画像のembeddingを生成"""
        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_embedding = self.model.encode_image(image)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            vector = image_embedding.cpu().numpy().tolist()[0]
            return np.concatenate([vector, np.zeros(1536-len(vector))]).tolist()
        except Exception as e:
            print(f"画像エンベディングエラー: {e}")
            return [0.0] * 1536

    def get_combined_embedding(self, metadata: Metadata) -> Dict[str, Any]:
        """本文と図の説明文を組み合わせた重み付きembeddingを生成"""
        main_text_embedding = np.array(self.get_embedding(metadata.content, 'main_text'))
        
        figure_embeddings = []
        for figure in metadata.figures.values():
            if figure.description:
                figure_embedding = np.array(self.get_embedding(figure.description, 'figure_desc'))
                figure_embeddings.append(figure_embedding)
        
        if figure_embeddings:
            figure_embedding_avg = np.mean(figure_embeddings, axis=0)
            total_embedding = np.concatenate([main_text_embedding, figure_embedding_avg])
        else:
            total_embedding = main_text_embedding
        
        return {
            "vector": total_embedding.tolist(),
            "metadata": {
                "type": "combined"
            }
        }

class TextEmbeddingProcessor:
    def __init__(self, client):
        self.client = client
        self.type_weights = {
            'main_text': 1.0,      # 本文
            'figure_desc': 0.6,    # 図の説明文
            'image': 0.6           # 画像
        }

    def get_embedding(self, text: str, text_type: str = 'main_text') -> List[float]:
        """テキストのembeddingを生成"""
        if isinstance(text, dict):
            text = ' '.join(str(v) for v in text.values())
        elif not isinstance(text, str):
            text = str(text)

        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            return self.normalize_embedding(embedding, text_type).tolist()
        except Exception as e:
            print(f"Embedding生成エラー: {e}")
            print(traceback.format_exc())
            return [0.0] * 1536

    def normalize_embedding(self, embedding: np.ndarray, text_type: str) -> np.ndarray:
        """埋め込みベクトルの正規化"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            normalized = embedding / norm
            return normalized * self.type_weights.get(text_type, 1.0)
        return embedding

    def get_combined_embedding(self, metadata: Metadata) -> Dict[str, Any]:
        """本文と図の説明文を組み合わせた重み付きembeddingを生成"""
        main_text_embedding = np.array(self.get_embedding(metadata.content, 'main_text'))
        
        figure_embeddings = []
        for figure in metadata.figures.values():
            if figure.description:
                figure_embedding = np.array(self.get_embedding(figure.description, 'figure_desc'))
                figure_embeddings.append(figure_embedding)
        
        if figure_embeddings:
            figure_embedding_avg = np.mean(figure_embeddings, axis=0)
            total_embedding = np.concatenate([main_text_embedding, figure_embedding_avg])
        else:
            total_embedding = main_text_embedding
        
        return {
            "vector": total_embedding.tolist(),
            "metadata": {
                "type": "combined"
            }
        }

def process_and_store_embeddings(text_processor: TextProcessor, 
                               text_embedding_processor: TextEmbeddingProcessor,
                               image_processor: ImageProcessor,
                               index,
                               file_path: str,
                               batch_size: int = 100):
    """バッチ処理を導入した保存処理"""
    metadata = text_processor.process_file(file_path)
    vectors_to_upsert = []
    
    try:
        # メインテキストの処理
        main_embedding = text_embedding_processor.get_combined_embedding(metadata)
        vectors_to_upsert.append({
            "id": f"{metadata.title}_main",
            "values": main_embedding["vector"],
            "metadata": {
                "title": metadata.title,
                "topic": metadata.topic,
                "type": "main_text",
                "content": metadata.content[:1000]
            }
        })
        
        # 図の処理
        for fig_id, figure in metadata.figures.items():
            if figure.description:
                vectors_to_upsert.append({
                    "id": f"{metadata.title}_{fig_id}_desc",
                    "values": text_embedding_processor.get_embedding(figure.description, 'figure_desc'),
                    "metadata": {
                        "title": metadata.title,
                        "figure_id": fig_id,
                        "type": "figure_desc",
                        "content": figure.description[:1000]
                    }
                })
        
        # バッチ単位でアップサート
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"バッチ {i//batch_size + 1} をアップサート完了")
        
        return True
    
    except Exception as e:
        print(f"データ保存エラー: {e}")
        print(traceback.format_exc())
        return False

def search_with_weights(query: str,
                       text_embedding_processor: TextEmbeddingProcessor,
                       index,
                       top_k: int = 10,
                       type_filter: Optional[str] = None,
                       min_score: float = 0.3) -> List[Dict]:
    """改善された重み付き検索"""
    try:
        query_embedding = text_embedding_processor.get_embedding(query, 'main_text')
        
        # 検索フィルターの設定
        filter_dict = {}
        if type_filter:
            filter_dict["type"] = type_filter
        
        # Pineconeで検索
        results = index.query(
            vector=query_embedding,
            top_k=top_k * 2,  # より多くの候補を取得
            filter=filter_dict if filter_dict else None,
            include_metadata=True
        )
        
        # スコアリングの改善
        processed_results = []
        for match in results.matches:
            if match.score >= min_score:
                processed_results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata,
                    "original_score": match.score
                })
        
        # スコアでソート
        return sorted(processed_results, key=lambda x: x["score"], reverse=True)[:top_k]
    
    except Exception as e:
        print(f"検索エラー: {e}")
        print(traceback.format_exc())
        return []

def validate_figure_data(text_content: str, figure_data: Dict[str, FigureData]) -> List[str]:
    """図の参照と実際のデータの整合性をチェック"""
    errors = []
    
    try:
        # 本文中の参照と実際の図の整合性チェック
        text_refs = set(re.findall(r'\(Fig\d+[a-z]?\)', text_content))
        figure_refs = set(figure_data.keys())
        
        # 参照されているが存在しない図
        missing_figures = {ref[1:-1] for ref in text_refs} - figure_refs
        if missing_figures:
            errors.append(f"Missing figures: {missing_figures}")
        
        # 存在するが参照されていない図
        unused_figures = figure_refs - {ref[1:-1] for ref in text_refs}
        if unused_figures:
            errors.append(f"Unused figures: {unused_figures}")
        
        return errors
    
    except Exception as e:
        print(f"データ検証エラー: {e}")
        print(traceback.format_exc())
        return [f"Validation error: {str(e)}"]

def initialize_services():
    """サービスの初期化"""
    try:
        env_vars = validate_environment()
        client = openai.OpenAI(api_key=env_vars["OPENAI_API_KEY"])
        pc = Pinecone(api_key=env_vars["PINECONE_API_KEY"])
        index = pc.Index("raiden02")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        returned_values = open_clip.create_model_and_transforms(
            "ViT-B/32",
            pretrained="openai",
            jit=True,
            force_quick_gelu=True
        )
        
        if len(returned_values) == 2:
            model, preprocess = returned_values
        elif len(returned_values) == 3:
            model, preprocess, _ = returned_values
        
        return client, index, model, preprocess, device
    
    except Exception as e:
        print(f"初期化エラー: {e}")
        print(traceback.format_exc())
        raise

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