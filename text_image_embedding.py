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

def extract_fig_and_case_texts(file_path):
    """テキストから [FigX] と [caseX] の説明を抽出し、ファイル名を考慮したIDを生成"""
    try:
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # ファイル名からページ範囲までを抽出
        base_parts = base_filename.split('_')
        cleaned_filename = '_'.join(part for part in base_parts if not part.isdigit() or '-' in part)
        
        with open(file_path, "r", encoding="utf-8") as f:
            text_data = f.read()
        
        result = {}
        
        # Fig の抽出
        fig_pattern = r"\[Fig(\d+[a-z]?)\]\s*(.*?)(?=\n\[(?:Fig|case)|$)"
        fig_matches = re.findall(fig_pattern, text_data, re.DOTALL)
        
        for num, desc in fig_matches:
            # ベースIDを生成
            base_id = f"{cleaned_filename}_Fig{num}"
            # テキストとイメージのIDを生成
            text_id = f"{base_id}_text"
            image_id = f"{base_id}_image"
            
            result[base_id] = {
                "type": "Fig",
                "number": num,
                "text": desc.strip(),
                "text_id": text_id,
                "image_id": image_id
            }
        
        # case の抽出
        case_pattern = r"\[case(\d+[a-z]?)\]\s*(.*?)(?=\n\[(?:Fig|case)|$)"
        case_matches = re.findall(case_pattern, text_data, re.DOTALL)
        
        for num, desc in case_matches:
            # ベースIDを生成
            base_id = f"{cleaned_filename}_case{num}"
            # テキストとイメージのIDを生成
            text_id = f"{base_id}_text"
            image_id = f"{base_id}_image"
            
            result[base_id] = {
                "type": "case",
                "number": num,
                "text": desc.strip(),
                "text_id": text_id,
                "image_id": image_id
            }
            
        return result

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
        
        all_entries = {}  # Fig と case の両方を格納
        for file_path in txt_files:
            entries = extract_fig_and_case_texts(file_path)
            all_entries.update(entries)
            fig_count = sum(1 for k, v in entries.items() if v["type"] == "Fig")
            case_count = sum(1 for k, v in entries.items() if v["type"] == "case")
            print(f"{file_path}: {fig_count}個の [FigX] と {case_count}個の [caseX] 説明文を取得")

        # テキストと画像の詳細を収集（embedding計算なし）
        text_entries = {"Fig": {}, "case": {}}  # {type: {base_number: set(variations)}}
        image_entries = {"Fig": {}, "case": {}}  # {type: {base_number: set(variations)}}
        missing_images = []
        entries_to_process = []  # embeddingを計算するデータを保存

        for base_id, data in all_entries.items():
            entry_type = data["type"]  # "Fig" または "case"
            number = data["number"]    # 番号（例: "1", "2a"）
            
            # 数字部分と変異部分を分離
            match = re.search(r'(\d+)([a-z]?)', number)
            if match:
                base_num = match.group(1)
                variation = match.group(2)
                
                # テキストエントリーを登録
                if base_num not in text_entries[entry_type]:
                    text_entries[entry_type][base_num] = set()
                if variation:
                    text_entries[entry_type][base_num].add(variation)
                elif not text_entries[entry_type][base_num]:
                    text_entries[entry_type][base_num].add('')
            
                # 画像の存在確認
                text_file_part = base_id.split(f'_{entry_type}')[0]
                image_path = f"data/{entry_type}{base_num}{variation}.jpg"
                
                if os.path.exists(image_path):
                    # 画像が存在する場合
                    if base_num not in image_entries[entry_type]:
                        image_entries[entry_type][base_num] = set()
                    if variation:
                        image_entries[entry_type][base_num].add(variation)
                    elif not image_entries[entry_type][base_num]:
                        image_entries[entry_type][base_num].add('')
                    
                    # embeddingを後で計算するためのデータを保存
                    entries_to_process.append({
                        "base_id": base_id,
                        "text_data": data,
                        "image_path": image_path,
                        "text_file_part": text_file_part,
                        "entry_type": entry_type,
                        "base_num": base_num,
                        "variation": variation
                    })
                else:
                    missing_images.append(image_path)
            
            print(f"- {base_id}の確認完了")

        # テキストと画像の詳細を表示
        total_fig_text = sum(1 for _, v in all_entries.items() if v["type"] == "Fig")
        total_case_text = sum(1 for _, v in all_entries.items() if v["type"] == "case")
        total_fig_image = sum(1 for item in entries_to_process if item["entry_type"] == "Fig")
        total_case_image = sum(1 for item in entries_to_process if item["entry_type"] == "case")
        
        print(f"\n確認したテキスト数: Fig={total_fig_text}, case={total_case_text}")
        print(f"確認した画像数: Fig={total_fig_image}, case={total_case_image}")
        
        # Fig と case それぞれの詳細を表示
        for entry_type in ["Fig", "case"]:
            print(f"\n=== {entry_type}の詳細 ===")
            all_base_numbers = sorted(set(text_entries[entry_type].keys()) | set(image_entries[entry_type].keys()))
            
            print(f"\nテキストの{entry_type}:")
            for base_num in all_base_numbers:
                variations = text_entries[entry_type].get(base_num, set())
                if variations:
                    var_str = ', '.join([f'{entry_type}{base_num}{v}' if v else f'{entry_type}{base_num}' for v in sorted(variations)])
                    print(f"{entry_type}{base_num}: {var_str}")
                else:
                    print(f"{entry_type}{base_num}: なし")

            print(f"\n画像の{entry_type}:")
            for base_num in all_base_numbers:
                variations = image_entries[entry_type].get(base_num, set())
                if variations:
                    var_str = ', '.join([f'{entry_type}{base_num}{v}' if v else f'{entry_type}{base_num}' for v in sorted(variations)])
                    print(f"{entry_type}{base_num}: {var_str}")
                else:
                    print(f"{entry_type}{base_num}: なし")

            # 不一致の確認
            has_mismatch = False
            print(f"\n=== {entry_type}の不一致確認 ===")
            for base_num in all_base_numbers:
                text_vars = text_entries[entry_type].get(base_num, set())
                image_vars = image_entries[entry_type].get(base_num, set())
                if text_vars != image_vars:
                    has_mismatch = True
                    print(f"\n{entry_type}{base_num}の不一致:")
                    if text_vars - image_vars:
                        print(f"  テキストにあって画像にない: {', '.join([f'{entry_type}{base_num}{v}' if v else f'{entry_type}{base_num}' for v in sorted(text_vars - image_vars)])}")
                    if image_vars - text_vars:
                        print(f"  画像にあってテキストにない: {', '.join([f'{entry_type}{base_num}{v}' if v else f'{entry_type}{base_num}' for v in sorted(image_vars - text_vars)])}")

            if not has_mismatch:
                print(f"\nすべての{entry_type}とそのバリエーションが一致しています。")

        if missing_images:
            print("\n見つからなかった画像ファイル:")
            for path in missing_images:
                print(f"- {path}")

        # ユーザー確認 - embeddingを行うかどうか
        proceed = input("\nembedding計算を開始しPineconeにデータを送信しますか？ (y/n): ").lower()
        if proceed != 'y':
            print("処理を中止します")
            return

        # y/n確認後にembedding計算を開始
        print("\nembedding計算を開始します...")
        vectors_to_upsert = []
        text_count = 0
        image_count = 0

        for item in entries_to_process:
            base_id = item["base_id"]
            data = item["text_data"]
            image_path = item["image_path"]
            text_file_part = item["text_file_part"]
            entry_type = item["entry_type"]
            base_num = item["base_num"]
            variation = item["variation"]
            
            # テキストのembedding取得
            print(f"テキストのembedding計算中: {data['text_id']}")
            text_emb = get_text_embedding(data["text"], client)
            text_count += 1
            vectors_to_upsert.append((
                data["text_id"],
                text_emb,
                {
                    "category": "dental",
                    "data_type": "text",
                    "text": data["text"],
                    "entry_type": entry_type,  # "Fig" または "case"
                    "related_image_id": data["image_id"]
                }
            ))
            
            # 画像のembedding取得
            print(f"画像のembedding計算中: {image_path}")
            image_result = get_image_embedding(image_path, model, preprocess, device)
            if image_result["status"] == "success":
                image_count += 1
                image_emb = image_result["vector"]
                image_id = f"{text_file_part}_{entry_type}{base_num}{variation}_image"
                vectors_to_upsert.append((
                    image_id,
                    image_emb,
                    {
                        "category": "dental",
                        "data_type": "image",
                        "text": data["text"],
                        "entry_type": entry_type,  # "Fig" または "case"
                        "related_text_id": data["text_id"]
                    }
                ))
            
            print(f"- {base_id}の処理完了")

        print(f"\n処理したテキスト数: {text_count}")
        print(f"処理した画像数: {image_count}")

        # Pineconeへのアップロード
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