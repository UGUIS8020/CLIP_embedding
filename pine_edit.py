#pineconeに保存されたベクトルデータimage,figure_descriptionのweightを0.75に更新するスクリプトです。

import os
from dotenv import load_dotenv
from pinecone import Pinecone
import argparse
from math import ceil

PINECONE_INDEX ="raiden" 

def display_page(vector_ids, page_size, page_number):
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

def get_vector_ids_and_update_weight(page_size=50, page_number=1):
    load_dotenv()
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(PINECONE_INDEX)

    stats = index.describe_index_stats()
    total_vector_count = stats.total_vector_count
    dimension = stats.dimension
    
    print(f"インデックスの統計情報:")
    print(f"総ベクトル数: {total_vector_count}")
    print(f"ベクトルの次元数: {dimension}")
    current_size_mb = (total_vector_count * dimension * 4) / (1024*1024)
    max_size_mb = 2048
    remaining_ratio = max_size_mb / current_size_mb
    
    print(f"推定使用容量: {current_size_mb:.2f} MB")
    print(f"最大容量: {max_size_mb} MB (2GB)")
    print(f"現在の使用量で約{remaining_ratio:.1f}倍のデータを保存可能\n")
    
    vector_ids = []
    try:
        list_result = index.list(prefix="", limit=total_vector_count)
        vector_ids = list(list_result.vectors.keys())
        print(f"取得したベクトルID数: {len(vector_ids)}")
    except Exception as e:
        print(f"リスト方式でエラー発生: {e}")
        print("代替方法でベクトルIDを取得します...")
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
    
    vector_ids.sort()
    total_pages = display_page(vector_ids, page_size, page_number)

    while True:
        print("\nコマンド:")
        print("- 次のページを表示: n または next")
        print("- 前のページを表示: p または prev")
        print("- 特定のページに移動: 1-{} の数字".format(total_pages))
        print("- 終了して更新処理へ進む: q または quit")
        
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
    
    # 更新処理スタート
    print("\n=== type='image' または 'figure_description' のベクトルを weight=0.75 に更新します ===")
    batch_size = 100
    ids_to_update = []

    for i in range(0, len(vector_ids), batch_size):
        batch_ids = vector_ids[i:i+batch_size]
        try:
            fetched = index.fetch(ids=batch_ids)
            for vid, vec in fetched.vectors.items():
                metadata = vec.metadata
                vec_type = metadata.get("type")
                if vec_type in ["image", "figure_description"]:
                    ids_to_update.append(vid)
        except Exception as e:
            print(f"フェッチ失敗（{i}-{i+batch_size}）: {e}")

    print(f"\n更新対象: {len(ids_to_update)} 件\n")

    for vid in ids_to_update:
        try:
            index.update(id=vid, set_metadata={"weight": 0.75})
            print(f"✅ {vid}: weight → 0.75 に更新完了")
        except Exception as e:
            print(f"❌ {vid}: 更新失敗 - {e}")

    print("\n✅ 全更新処理が完了しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PineconeのベクトルIDを表示し、weight=0.75を0.7に更新します')
    parser.add_argument('--page-size', type=int, default=300, help='1ページあたりの表示件数（デフォルト: 300）')
    parser.add_argument('--page', type=int, default=1, help='表示を開始するページ番号（デフォルト: 1）')
    args = parser.parse_args()
    
    get_vector_ids_and_update_weight(args.page_size, args.page)
