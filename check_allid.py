import os
from dotenv import load_dotenv
from pinecone import Pinecone

def get_all_vector_ids():
    try:
        # 環境変数の読み込み
        load_dotenv()
        
        # Pineconeの初期化
        pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY')
        )
        
        # インデックスの取得
        index = pc.Index("raiden")
        
        # クエリを使用してすべてのベクトルを取得
        # ダミーのベクトルを使用して全件取得
        dummy_vector = [0.0] * 1536  # OpenAIのembeddingサイズ
        query_response = index.query(
            vector=dummy_vector,
            top_k=10000,  # 取得する最大数（必要に応じて調整）
            include_metadata=True
        )
        
        # IDを抽出
        all_ids = [match.id for match in query_response.matches]
        
        print(f"取得したベクトルID数: {len(all_ids)}")
        print("\nベクトルID一覧:")
        for vector_id in all_ids:
            print(vector_id)
            
        return all_ids
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

if __name__ == "__main__":
    get_all_vector_ids()