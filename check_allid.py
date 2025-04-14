import os
from dotenv import load_dotenv
from pinecone import Pinecone
import argparse

def display_page(vector_infos, page_size, page_number):
    """指定されたページのベクトルIDとweightを表示"""
    total_pages = (len(vector_infos) + page_size - 1) // page_size
    start_idx = (page_number - 1) * page_size
    end_idx = min(start_idx + page_size, len(vector_infos))

    print(f"\nページ {page_number}/{total_pages}")
    print(f"表示範囲: {start_idx + 1}～{end_idx} / 全{len(vector_infos)}件")
    print("\nベクトル情報一覧:")

    for i, info in enumerate(vector_infos[start_idx:end_idx], start=start_idx + 1):
        print(f"{i}. ID: {info['id']} | weight: {info['weight']}")

    print(f"\n--- ページ {page_number}/{total_pages} ---")
    return total_pages

def get_vector_infos_with_weight(page_size=50, page_number=1):
    """ベクトルIDとメタデータweightを取得し、ページ表示"""

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
    current_size_mb = (total_vector_count * dimension * 4) / (1024 * 1024)
    max_size_mb = 2048  # 2GB
    remaining_ratio = max_size_mb / current_size_mb

    print(f"推定使用容量: {current_size_mb:.2f} MB")
    print(f"最大容量: {max_size_mb} MB (2GB)")
    print(f"現在の使用量で約{remaining_ratio:.1f}倍のデータを保存可能\n")

    vector_infos = []
    try:
        # リストベクトル方式（メタデータは取得できないためスキップ）
        raise Exception("list() では weight を取得できないため query に切り替え")

    except Exception as e:
        print(f"ベクトル一覧取得エラーまたは制限: {e}")
        print("代替方法（query）でベクトル情報を取得します...")

        # 代替方法：ダミーベクトルでquery
        dummy_vector = [0.0] * dimension
        fetch_limit = min(10000, total_vector_count)

        query_response = index.query(
            vector=dummy_vector,
            top_k=fetch_limit,
            include_metadata=True,   # ← weight を取得
            include_values=False
        )

        vector_infos = [
            {
                "id": match.id,
                "weight": match.metadata.get("weight", "N/A")  # weightがない場合は"N/A"
            }
            for match in query_response.matches
        ]
        print(f"取得したベクトル数: {len(vector_infos)}")

    # IDでソート（任意）
    vector_infos.sort(key=lambda x: x["id"])

    # 指定されたページを表示
    total_pages = display_page(vector_infos, page_size, page_number)

    # ページングループ
    while True:
        print("\nコマンド:")
        print("- 次のページを表示: n または next")
        print("- 前のページを表示: p または prev")
        print(f"- 特定のページに移動: 1～{total_pages}")
        print("- 終了: q または quit")

        command = input("\n操作を入力してください: ").lower().strip()

        if command in ['q', 'quit']:
            break
        elif command in ['n', 'next']:
            if page_number < total_pages:
                page_number += 1
                display_page(vector_infos, page_size, page_number)
            else:
                print("これ以上次のページはありません。")
        elif command in ['p', 'prev']:
            if page_number > 1:
                page_number -= 1
                display_page(vector_infos, page_size, page_number)
            else:
                print("これ以上前のページはありません。")
        else:
            try:
                new_page = int(command)
                if 1 <= new_page <= total_pages:
                    page_number = new_page
                    display_page(vector_infos, page_size, page_number)
                else:
                    print(f"ページ番号は1から{total_pages}の間で指定してください。")
            except ValueError:
                print("無効なコマンドです。")

    return vector_infos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PineconeのベクトルIDとweightを表示します')
    parser.add_argument('--page-size', type=int, default=50, help='1ページあたりの表示件数（デフォルト: 50）')
    parser.add_argument('--page', type=int, default=1, help='表示を開始するページ番号（デフォルト: 1）')
    args = parser.parse_args()

    get_vector_infos_with_weight(args.page_size, args.page)