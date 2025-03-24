import os
import openai
from pinecone import Pinecone
from dotenv import load_dotenv
import glob

# 環境変数をロード
load_dotenv()

class TextProcessor:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index("raiden02")

    def get_embedding(self, text: str) -> list:
        """テキストのembeddingを生成"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding生成エラー: {e}")
            return None

    def process_file(self, file_path: str):
        """テキストファイルを処理"""
        try:
            # ファイルを読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # embeddingを生成
            embedding = self.get_embedding(text)
            if embedding:
                # ファイル名からIDを生成
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # ベクトルデータを準備
                vector_data = {
                    "id": f"{base_name}_content",
                    "values": embedding,
                    "metadata": {
                        "file_name": base_name,
                        "text": text,
                        "data_type": "content"
                    }
                }
                
                # Pineconeにアップロード
                self.index.upsert(vectors=[vector_data])
                print(f"  {base_name}のベクトルをPineconeに保存しました")

        except Exception as e:
            print(f"ファイル処理エラー: {e}")

def main():
    # TextProcessorのインスタンスを作成
    processor = TextProcessor()

    # テキストファイルを検索
    txt_files = glob.glob("data/*.txt")
    if not txt_files:
        print("data ディレクトリにtxtファイルが見つかりません")
        return

    print(f"{len(txt_files)}個のテキストファイルを処理します")
    
    # 各ファイルを処理
    for file_path in txt_files:
        print(f"\n{file_path}の処理を開始します...")
        processor.process_file(file_path)

if __name__ == "__main__":
    main() 