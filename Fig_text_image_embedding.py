import os
import re
import glob
import numpy as np
import openai
import pinecone
import torch
import open_clip
import traceback
from PIL import Image
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 環境変数をロード
load_dotenv()

def extract_fig_texts(file_path):
    """テキストから [FigX] の説明を抽出し、ファイル名を考慮したIDを生成"""
    try:
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # ファイル名からページ範囲までを抽出
        # 例: "Autologous_Tooth_Transplantation01_chapter01_007-011_016" から
        # "Autologous_Tooth_Transplantation01_chapter01_007-011" を取得
        base_parts = base_filename.split('_')
        cleaned_filename = '_'.join(part for part in base_parts if not part.isdigit() or '-' in part)
        
        with open(file_path, "r", encoding="utf-8") as f:
            text_data = f.read()
        
        pattern = r"\[Fig(\d+[a-z]?)\]\s*(.*?)(?=\n\[Fig|$)"
        matches = re.findall(pattern, text_data, re.DOTALL)
        
        result = {}
        for num, desc in matches:
            # ベースIDを生成
            base_id = f"{cleaned_filename}_Fig{num}"
            # テキストとイメージのIDを生成
            text_id = f"{base_id}_text"
            image_id = f"{base_id}_image"
            
            result[base_id] = {
                "text": desc.strip(),
                "text_id": text_id,
                "image_id": image_id
            }
            
        return result

    except Exception as e:
        print(f"テキスト抽出エラー: {e}")
        print(traceback.format_exc())
        return {}

    except Exception as e:
        print(f"テキスト抽出エラー: {e}")
        print(traceback.format_exc())
        return {}

    except Exception as e:
        print(f"テキスト抽出エラー: {e}")
        print(traceback.format_exc())
        return {}

    except Exception as e:
        print(f"テキスト抽出エラー: {e}")
        print(traceback.format_exc())
        return {}

def validate_environment():
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY")
    }
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise EnvironmentError(f"必要な環境変数が設定されていません: {', '.join(missing_vars)}")
    return required_vars

# 初期化処理をまとめる
def initialize_services():
    env_vars = validate_environment()
    client = openai.OpenAI(api_key=env_vars["OPENAI_API_KEY"])
    pc = Pinecone(api_key=env_vars["PINECONE_API_KEY"])
    index = pc.Index("raiden")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    returned_values = open_clip.create_model_and_transforms(
        "ViT-B/32",
        pretrained="openai",
        jit=True,  # JITコンパイルを有効化
        force_quick_gelu=True  # QuickGELUを強制的に有効化
    )
    
    if len(returned_values) == 2:
        model, preprocess = returned_values
    elif len(returned_values) == 3:
        model, preprocess, _ = returned_values
    else:
        raise ValueError("open_clip.create_model_and_transforms からの予期しない戻り値の数です")
    
    model.to(device)
    return client, index, model, preprocess, device

def get_image_embedding(image_path, model, preprocess, device):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = model.encode_image(image)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        # 512次元を1536次元に拡張
        vector = image_embedding.cpu().numpy().tolist()[0]
        expanded_vector = np.concatenate([vector, np.zeros(1536-len(vector))]).tolist()
        return {
            "vector": expanded_vector,
            "model": "open_clip",
            "status": "success"
        }
    except Exception as e:
        print(f"画像エンベディングエラー: {e}")
        return {
            "vector": [0.0] * 1536,  # 1536次元のゼロベクトル
            "model": "open_clip",
            "status": "error",
            "error_message": str(e)
        }

# 512次元ベクトルを 1536次元に拡張（ゼロ埋め）
def expand_embedding(embedding_dict, target_dim=1536):
    vector = embedding_dict["vector"]
    extra_dims = target_dim - len(vector)
    expanded_vector = np.concatenate([vector, np.zeros(extra_dims)]).tolist()
    
    return {
        "vector": expanded_vector,
        "original_dim": len(vector),
        "model": embedding_dict["model"],
        "status": embedding_dict["status"]
    }

def get_text_embedding(text, client, chunk_size=1000, chunk_overlap=200):
    if isinstance(text, dict):
        text = ' '.join(str(v) for v in text.values())
    elif not isinstance(text, str):
        text = str(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    # 逐次的に平均を計算
    embedding_sum = np.zeros(1536)
    count = 0
    
    for chunk in chunks:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            embedding_sum += np.array(response.data[0].embedding)
            count += 1
        except Exception as e:
            print(f"Embedding生成エラー: {e}")
            print(traceback.format_exc())
    
    return (embedding_sum / count if count > 0 else embedding_sum).tolist()

def main():
    try:
        client, index, model, preprocess, device = initialize_services()
        
        txt_files = glob.glob("data/*.txt")
        if not txt_files:
            raise FileNotFoundError("data ディレクトリにtxtファイルが見つかりません")
        
        all_fig_texts = {}
        for file_path in txt_files:
            fig_texts = extract_fig_texts(file_path)
            all_fig_texts.update(fig_texts)
            print(f"{file_path}: {len(fig_texts)}個の [FigX] 説明文を取得")

        vectors_to_upsert = []
        missing_images = []
        text_count = 0
        image_count = 0
        text_figs = {}  # {base_number: set(variations)}
        image_figs = {}  # {base_number: set(variations)}

        for base_id, data in all_fig_texts.items():
            # テキストのFig番号を抽出
            text_fig_match = re.search(r'Fig(\d+)([a-z]?)', base_id)
            if text_fig_match:
                base_num = text_fig_match.group(1)
                variation = text_fig_match.group(2)
                if base_num not in text_figs:
                    text_figs[base_num] = set()
                if variation:
                    text_figs[base_num].add(variation)
                elif not text_figs[base_num]:  # バリエーションがない場合は空文字を追加
                    text_figs[base_num].add('')
            
            # テキストのembedding取得
            text_emb = get_text_embedding(data["text"], client)
            text_count += 1
            vectors_to_upsert.append((
                data["text_id"],
                text_emb,
                {
                    "category": "dental",
                    "data_type": "text",
                    "text": data["text"],
                    "related_image_id": data["image_id"]
                }
            ))
            
            # 画像の処理
            fig_match = re.search(r'Fig(\d+)([a-z]?)', base_id)
            if fig_match:
                base_num = fig_match.group(1)
                variation = fig_match.group(2)
                text_file_part = base_id.split('_Fig')[0]  # base_idからtext_file_partを抽出
                image_path = f"data/Fig{base_num}{variation}.jpg"
                
                if os.path.exists(image_path):
                    image_result = get_image_embedding(image_path, model, preprocess, device)
                    if image_result["status"] == "success":
                        image_count += 1
                        if base_num not in image_figs:
                            image_figs[base_num] = set()
                        if variation:
                            image_figs[base_num].add(variation)
                        elif not image_figs[base_num]:  # バリエーションがない場合は空文字を追加
                            image_figs[base_num].add('')
                            
                        image_emb = image_result["vector"]
                        image_id = f"{text_file_part}_Fig{base_num}{variation}_image"
                        vectors_to_upsert.append((
                            image_id,
                            image_emb,
                            {
                                "category": "dental",
                                "data_type": "image",
                                "text": data["text"],
                                "related_text_id": data["text_id"]
                            }
                        ))
                else:
                    missing_images.append(image_path)
            print(f"- {base_id}の処理完了")

        # テキストと画像の詳細を表示
        print(f"\nテキスト数: {text_count}")
        print(f"画像数: {image_count}")
        
        print("\n=== Figの詳細 ===")
        all_base_numbers = sorted(set(text_figs.keys()) | set(image_figs.keys()))
        
        print("\nテキストのFig:")
        for base_num in all_base_numbers:
            variations = text_figs.get(base_num, set())
            if variations:
                var_str = ', '.join([f'Fig{base_num}{v}' if v else f'Fig{base_num}' for v in sorted(variations)])
                print(f"Fig{base_num}: {var_str}")
            else:
                print(f"Fig{base_num}: なし")

        print("\n画像のFig:")
        for base_num in all_base_numbers:
            variations = image_figs.get(base_num, set())
            if variations:
                var_str = ', '.join([f'Fig{base_num}{v}' if v else f'Fig{base_num}' for v in sorted(variations)])
                print(f"Fig{base_num}: {var_str}")
            else:
                print(f"Fig{base_num}: なし")

        # 不一致の確認
        has_mismatch = False
        print("\n=== 不一致の確認 ===")
        for base_num in all_base_numbers:
            text_vars = text_figs.get(base_num, set())
            image_vars = image_figs.get(base_num, set())
            if text_vars != image_vars:
                has_mismatch = True
                print(f"\nFig{base_num}の不一致:")
                if text_vars - image_vars:
                    print(f"  テキストにあって画像にない: {', '.join([f'Fig{base_num}{v}' if v else f'Fig{base_num}' for v in sorted(text_vars - image_vars)])}")
                if image_vars - text_vars:
                    print(f"  画像にあってテキストにない: {', '.join([f'Fig{base_num}{v}' if v else f'Fig{base_num}' for v in sorted(image_vars - text_vars)])}")

        if not has_mismatch:
            print("\nすべてのFigとそのバリエーションが一致しています。")

        if missing_images:
            print("\n見つからなかった画像ファイル:")
            for path in missing_images:
                print(f"- {path}")

        proceed = input("\nPineconeにデータを送信しますか？ (y/n): ").lower()
        if proceed != 'y':
            print("処理を中止します")
            return

        if vectors_to_upsert:
            try:
                index.upsert(vectors=vectors_to_upsert)
                print(f"\n{len(vectors_to_upsert)}個のベクトルをPineconeに保存しました")
            except Exception as e:
                print(f"\nPineconeへの保存中にエラーが発生: {e}")
        else:
            print("\nアップロードするデータがありません")

    except Exception as e:
        print(f"\n処理中にエラーが発生: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()