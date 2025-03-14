import os
from dotenv import load_dotenv
from pinecone import Pinecone

def get_vector_ids_fast_with_storage_info():
    # 環境変数の読み込み
    load_dotenv()
    
    # Pineconeの初期化
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
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
    
    # 全てのベクトルIDを直接取得する
    # describe_index_stats()のnamespaceから情報を取得
    namespaces = stats.namespaces
    
    # まずはリスト方式で試す
    try:
        # リストベクトル方式
        list_result = index.list(prefix="", limit=total_vector_count)
        vector_ids = list_result.vectors.keys()
        
        print(f"\n取得したベクトルID数: {len(vector_ids)}")
        print("\nベクトルID一覧:")
        for vector_id in vector_ids:
            print(vector_id)
        
        return list(vector_ids)
    
    except Exception as e:
        print(f"リスト方式でエラー発生: {e}")
        print("代替方法でベクトルIDを取得します...")
        
        # 代替方法: シンプルなクエリをランキングスコア無視で実行
        dummy_vector = [0.0] * dimension
        
        # 一度に全てのベクトルを取得（制限内であれば）
        fetch_limit = min(10000, total_vector_count)
        
        query_response = index.query(
            vector=dummy_vector,
            top_k=fetch_limit,
            include_metadata=False,
            include_values=False
        )
        
        # IDを抽出
        all_ids = [match.id for match in query_response.matches]
        
        print(f"\n取得したベクトルID数: {len(all_ids)}")
        print("\nベクトルID一覧:")
        for vector_id in all_ids:
            print(vector_id)
        
        return all_ids

if __name__ == "__main__":
    get_vector_ids_fast_with_storage_info()