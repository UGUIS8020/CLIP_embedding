import os
from pinecone import Pinecone
from dotenv import load_dotenv

# 環境変数のロード
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Pinecone 初期化
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("raiden02")  # インデックスの参照

def delete_all_pinecone_data():
    """Pinecone 内の全データを削除"""
    try:
        # 削除前のデータ数を確認
        stats_before = index.describe_index_stats()
        total_vectors = stats_before["total_vector_count"]
        print(f"\n削除前のベクトル数: {total_vectors}")

        if total_vectors == 0:
            print("データは既に空です。削除の必要はありません。")
            return

        # データ削除
        print("\nPinecone の全データを削除中...")
        index.delete(delete_all=True)

        # 削除後のデータ数を確認
        stats_after = index.describe_index_stats()
        total_vectors_after = stats_after["total_vector_count"]

        if total_vectors_after == 0:
            print("✅ すべてのデータを削除しました。")
        else:
            print(f"⚠️ 削除後も {total_vectors_after} 個のベクトルが残っています。")

    except Exception as e:
        print(f"⚠️ データ削除中にエラーが発生しました: {e}")

# 実行
if __name__ == "__main__":
    delete_all_pinecone_data()