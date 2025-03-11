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
        
        # インデックスの統計情報を取得
        stats = index.describe_index_stats()
        total_vector_count = stats.total_vector_count
        dimension = stats.dimension
        
        print(f"インデックスの統計情報:")
        print(f"総ベクトル数: {total_vector_count}")
        print(f"ベクトルの次元数: {dimension}")
        current_size_mb = (total_vector_count * dimension * 4) / (1024*1024)
        max_size_mb = 2048  # 2GB
        remaining_ratio = max_size_mb / current_size_mb
        
        print(f"推定使用容量: {current_size_mb:.2f} MB")
        print(f"最大容量: {max_size_mb} MB (2GB)")
        print(f"現在の使用量で約{remaining_ratio:.1f}倍のデータを保存可能\n")
        
        # 全てのベクトルIDを格納するリスト
        all_ids = []
        
        # ページネーションで全件取得
        batch_size = 10000  # 1回のクエリで取得する件数
        dummy_vector = [0.0] * dimension
        
        while True:
            query_response = index.query(
                vector=dummy_vector,
                top_k=batch_size,
                include_metadata=True,
                filter={
                    "id": {"$nin": all_ids}  # 既に取得したIDを除外
                } if all_ids else None
            )
            
            # 新しいIDを追加
            new_ids = [match.id for match in query_response.matches]
            if not new_ids:  # 新しいIDがない場合は終了
                break
                
            all_ids.extend(new_ids)
            print(f"進捗: {len(all_ids)}/{total_vector_count}件取得済み")
        
        print(f"\n取得したベクトルID数: {len(all_ids)}")
        print("\nベクトルID一覧:")
        for vector_id in all_ids:
            print(vector_id)
            
        return all_ids
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

if __name__ == "__main__":
    get_all_vector_ids()