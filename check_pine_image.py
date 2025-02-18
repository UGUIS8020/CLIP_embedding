import os
from pinecone import Pinecone
from dotenv import load_dotenv

# 環境変数のロード
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Pinecone 初期化
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("text-search")

# インデックスに保存されているデータを取得
response = index.describe_index_stats()
print("✅ Pinecone インデックス情報:", response)

# 画像IDリスト（例: すべての画像を取得するなら、別の方法が必要）
image_ids = ["Fig1.jpg", "Fig2.jpg", "Fig3.jpg", "Fig4.jpg"]  # ここは適宜修正

print("\n✅ Pinecone に保存されている画像データ一覧:")
for image_id in image_ids:
    fetch_response = index.fetch([image_id])

    if image_id in fetch_response["vectors"]:
        vector_data = fetch_response["vectors"][image_id]
        vector = vector_data.get("values", [])
        metadata = vector_data.get("metadata", {})

        print(f"- ID: {image_id}, 画像ファイル: {metadata.get('image', '不明')}, ベクトル次元: {len(vector)}")
    else:
        print(f"- ID: {image_id} は Pinecone に存在しません")