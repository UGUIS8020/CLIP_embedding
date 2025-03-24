import os
from dotenv import load_dotenv
from pinecone import Pinecone
import argparse

def display_page(vector_ids, page_size, page_number):
    """指定されたページのベクトルIDを表示"""
    total_pages = (len(vector_ids) + page_size - 1) // page_size
    start_idx = (page_number - 1) * page_size
    end_idx = min(start_idx + page_size, len(vector_ids))
    
    print(f"\nページ {page_number}/{total_pages}")
    print(f"表示範囲: {start_idx + 1}～{end_idx} / 全{len(vector_ids)}件")
    print("\nベクトルID一覧:")
    
    for i, vector_id in enumerate(vector_ids[start_idx:end_idx], start=start_idx + 1):
        print(f"{i}. {vector_id}")
    
    print(f"\n--- ページ {page_number}/{total_pages} ---")
    return total_pages

def get_vector_ids_fast_with_storage_info(page_size=50, page_number=1):
    """ベクトルIDを取得し、ページネーション形式で表示"""
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
    
    vector_ids = []
    try:
        # リストベクトル方式
        list_result = index.list(prefix="", limit=total_vector_count)
        vector_ids = list(list_result.vectors.keys())
        print(f"取得したベクトルID数: {len(vector_ids)}")
        
    except Exception as e:
        print(f"リスト方式でエラー発生: {e}")
        print("代替方法でベクトルIDを取得します...")
        
        # 代替方法: シンプルなクエリをランキングスコア無視で実行
        dummy_vector = [0.0] * dimension
        fetch_limit = min(10000, total_vector_count)
        
        query_response = index.query(
            vector=dummy_vector,
            top_k=fetch_limit,
            include_metadata=False,
            include_values=False
        )
        
        vector_ids = [match.id for match in query_response.matches]
        print(f"取得したベクトルID数: {len(vector_ids)}")
    
    # ソートしてから表示（オプション）
    vector_ids.sort()
    
    # 指定されたページを表示
    total_pages = display_page(vector_ids, page_size, page_number)
    
    while True:
        print("\nコマンド:")
        print("- 次のページを表示: n または next")
        print("- 前のページを表示: p または prev")
        print("- 特定のページに移動: 1-{} の数字".format(total_pages))
        print("- 終了: q または quit")
        
        command = input("\n操作を入力してください: ").lower().strip()
        
        if command in ['q', 'quit']:
            break
        elif command in ['n', 'next']:
            if page_number < total_pages:
                page_number += 1
                display_page(vector_ids, page_size, page_number)
        elif command in ['p', 'prev']:
            if page_number > 1:
                page_number -= 1
                display_page(vector_ids, page_size, page_number)
        else:
            try:
                new_page = int(command)
                if 1 <= new_page <= total_pages:
                    page_number = new_page
                    display_page(vector_ids, page_size, page_number)
                else:
                    print(f"ページ番号は1から{total_pages}の間で指定してください。")
            except ValueError:
                print("無効なコマンドです。")
    
    return vector_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PineconeのベクトルIDを表示します')
    parser.add_argument('--page-size', type=int, default=300, help='1ページあたりの表示件数（デフォルト: 300）')
    parser.add_argument('--page', type=int, default=1, help='表示を開始するページ番号（デフォルト: 1）')
    args = parser.parse_args()
    
    get_vector_ids_fast_with_storage_info(args.page_size, args.page)