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

# 本文とfigテキスト、fig画像全てを一回でembeddingする。
# summeryを生成する。
# 01,02の進化版

# 環境変数をロード
load_dotenv()

def extract_metadata_and_texts(file_path):
    """テキストからメタデータおよび[FigX]と[caseX]の説明を抽出し、ファイル名を考慮したIDを生成"""
    try:
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # ファイル名からページ範囲までを抽出
        base_parts = base_filename.split('_')
        cleaned_filename = '_'.join(part for part in base_parts if not part.isdigit() or '-' in part)
        
        with open(file_path, "r", encoding="utf-8") as f:
            text_data = f.read()
        
        result = {}
        
        # メタデータの抽出
        metadata = {
            "title": None,                       
            "item": None,
            "content": None,
            "figure_contexts": {},
            "grouped_figure_contexts": {}
        }
        
        # メタデータの正規表現パターン
        title_pattern = r"title\[(.*?)\]"           
        item_pattern = r"item\[(.*?)\]"
        
        # メタデータの抽出
        title_match = re.search(title_pattern, text_data)
        if title_match:
            metadata["title"] = title_match.group(1)              
            
        item_match = re.search(item_pattern, text_data)
        if item_match:
            metadata["item"] = item_match.group(1)
        
        # 本文の抽出（item[] の後から次のtitle[] までの内容）
        if item_match:
            item_end = item_match.end()
            next_title_match = re.search(title_pattern, text_data[item_end:])
            if next_title_match:
                metadata["content"] = text_data[item_end:item_end + next_title_match.start()].strip()
            else:
                metadata["content"] = text_data[item_end:].strip()
        
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
                "image_id": image_id,
                "metadata": metadata  # メタデータを追加
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
                "image_id": image_id,
                "metadata": metadata  # メタデータを追加
            }
            
        # 本文から各図への参照コンテキストを抽出
        if metadata["content"]:
            # 本文中に現れる全てのFigとcaseの参照を検出
            fig_refs = set()
            case_refs = set()
            
            # Fig参照を検出（Fig1, Fig2a, Fig.3 などのパターン）
            fig_patterns = [r'Fig\.?\s*(\d+[a-z]?)', r'\(Fig\.?\s*(\d+[a-z]?)\)']
            for pattern in fig_patterns:
                for match in re.finditer(pattern, metadata["content"]):
                    fig_num = match.group(1)
                    fig_refs.add(f"Fig{fig_num}")
            
            # case参照を検出
            case_patterns = [r'case\s*(\d+[a-z]?)', r'\(case\s*(\d+[a-z]?)\)']
            for pattern in case_patterns:
                for match in re.finditer(pattern, metadata["content"]):
                    case_num = match.group(1)
                    case_refs.add(f"case{case_num}")
            
            # 各参照のコンテキストを抽出
            for ref_id in fig_refs:
                context = extract_reference_context(metadata["content"], ref_id)
                if context:
                    metadata["figure_contexts"][ref_id] = context
                    
            for ref_id in case_refs:
                context = extract_reference_context(metadata["content"], ref_id)
                if context:
                    metadata["figure_contexts"][ref_id] = context
        
        # 抽出したコンテキストをグループ化
        if metadata["figure_contexts"]:
            metadata["grouped_figure_contexts"] = group_figures_and_contexts(metadata["figure_contexts"])
        
        return result, metadata

    except Exception as e:
        print(f"テキスト抽出エラー: {e}")
        print(traceback.format_exc())
        return {}, {}

def extract_reference_context(content, ref_id, context_window=200):
    """
    本文中から特定の参照（Fig1など）が言及されている周辺テキストを抽出
    
    Args:
        content: 本文テキスト
        ref_id: 検索する参照ID（'Fig1'など）
        context_window: 参照前後の抽出する文字数
    
    Returns:
        抽出されたコンテキストテキスト
    """
    if not content or not ref_id:
        return ""
    
    try:
        # 参照パターンの正規表現（Fig1, Fig1a, Fig.1, (Fig1), Fig1,2 などのパターンに対応）
        patterns = [
            r'{}[^a-zA-Z0-9]'.format(ref_id),  # Fig1,
            r'{}\b'.format(ref_id),            # Fig1
            r'\({}\)'.format(ref_id),          # (Fig1)
            r'{}\.'.format(ref_id),            # Fig1.
        ]
        
        contexts = []
        for pattern in patterns:
            matches = list(re.finditer(pattern, content))
            for match in matches:
                start_pos = max(0, match.start() - context_window)
                end_pos = min(len(content), match.end() + context_window)
                
                # 文の開始と終了位置を調整（できるだけ文単位で抽出）
                while start_pos > 0 and content[start_pos] not in "。.":
                    start_pos -= 1
                
                while end_pos < len(content) - 1 and content[end_pos] not in "。.":
                    end_pos += 1
                
                context = content[start_pos:end_pos].strip()
                if context:
                    contexts.append(context)
        
        # 重複するコンテキストを除去
        unique_contexts = []
        for ctx in contexts:
            if not any(ctx in uc for uc in unique_contexts):
                unique_contexts.append(ctx)
        
        return "\n".join(unique_contexts)
    
    except Exception as e:
        print(f"参照コンテキスト抽出エラー: {e}")
        return ""

def group_figures_and_contexts(figure_contexts):
    """
    図の参照をグループ化し、関連するコンテキストをまとめる
    
    Args:
        figure_contexts: {図ID: コンテキスト} の辞書
    
    Returns:
        {ベース図ID: {コンテキスト, 含まれる図IDs}} の辞書
    """
    # グループ化された結果を格納する辞書
    grouped_contexts = {}
    
    # 正規表現でFigやcaseの基本IDを抽出（例: Fig1a -> Fig1）
    pattern = r'(Fig|case)(\d+)[a-z]?'
    
    for fig_id, context in figure_contexts.items():
        match = re.match(pattern, fig_id)
        if match:
            prefix = match.group(1)  # "Fig" または "case"
            base_num = match.group(2)  # 数字部分
            base_id = f"{prefix}{base_num}"  # 基本ID (Fig1など)
            
            # グループがまだ存在しない場合は初期化
            if base_id not in grouped_contexts:
                grouped_contexts[base_id] = {
                    "contexts": [],
                    "included_ids": []
                }
            
            # コンテキストとIDを追加
            if context not in grouped_contexts[base_id]["contexts"]:
                grouped_contexts[base_id]["contexts"].append(context)
            
            grouped_contexts[base_id]["included_ids"].append(fig_id)
    
    # 各グループのコンテキストを結合
    for base_id, group_data in grouped_contexts.items():
        # 重複を除去し、コンテキストを結合
        unique_contexts = []
        for ctx in group_data["contexts"]:
            if not any(ctx in uc for uc in unique_contexts):
                unique_contexts.append(ctx)
        
        group_data["combined_context"] = "\n".join(unique_contexts)
        # 含まれる図IDをソート
        group_data["included_ids"] = sorted(set(group_data["included_ids"]))
    
    return grouped_contexts

def generate_summary(content, client, max_tokens=150):
    """
    OpenAI APIを使用して本文の要約を生成する
    
    Args:
        content: 要約する本文
        client: OpenAIクライアントインスタンス
        max_tokens: 要約の最大トークン数
    
    Returns:
        生成された要約テキスト
    """
    if not content or len(content.strip()) < 50:
        return ""  # コンテンツが短すぎる場合は要約しない
    
    try:
        # コンテンツが長すぎる場合は最初の2000文字のみを使用
        truncated_content = content[:2000] if len(content) > 2000 else content
        
        # OpenAI APIを使用して要約を生成
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # または "gpt-4" などの適切なモデル
            messages=[
                {"role": "system", "content": "あなたは日本語テキストを要約する専門家です。与えられたテキストの重要なポイントを簡潔に要約してください。"},
                {"role": "user", "content": f"以下の歯科医学に関するテキストを100-150単語で要約してください：\n\n{truncated_content}"}
            ],
            max_tokens=max_tokens,
            temperature=0.5  # より決定論的な要約のため温度を低めに設定
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
    
    except Exception as e:
        print(f"要約生成エラー: {e}")
        return ""  # エラーが発生した場合は空の要約を返す

def generate_figure_group_summary(base_id, group_data, client, max_tokens=150):
    """
    図グループに特化した要約を生成
    
    Args:
        base_id: ベース図ID (例: "Fig1")
        group_data: グループデータ (コンテキストと含まれる図IDを含む)
        client: OpenAIクライアント
        max_tokens: 要約の最大トークン数
    
    Returns:
        図グループに特化した要約テキスト
    """
    if not group_data["combined_context"]:
        return ""
    
    try:
        # 含まれる図IDをカンマ区切りのリストに
        included_figs = ", ".join(group_data["included_ids"])
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "あなたは医学・歯科学の図表の役割を解説する専門家です。本文中で言及されている図表グループについて、その重要性と文脈を要約してください。"},
                {"role": "user", "content": f"以下の歯科医学の本文は{included_figs}について言及しています。これらの図が何を示していて、どのような重要性があるかを80-120字程度で要約してください：\n\n{group_data['combined_context']}"}
            ],
            max_tokens=max_tokens,
            temperature=0.4
        )
        
        group_summary = response.choices[0].message.content.strip()
        return group_summary
    
    except Exception as e:
        print(f"図グループ要約生成エラー: {e}")
        return ""

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
    index = pc.Index("e-sports")
    
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
        all_metadata = {}  # 各ファイルのメタデータを格納
        
        for file_path in txt_files:
            entries, metadata = extract_metadata_and_texts(file_path)
            all_entries.update(entries)
            all_metadata[file_path] = metadata
            
            fig_count = sum(1 for k, v in entries.items() if v["type"] == "Fig")
            case_count = sum(1 for k, v in entries.items() if v["type"] == "case")
            print(f"{file_path}: {fig_count}個の [FigX] と {case_count}個の [caseX] 説明文を取得")
            
            # メタデータの表示
            if metadata["title"] or metadata["item"]:
                print(f"  メタデータ: title={metadata['title']}, item={metadata['item']}")

            # contentがある場合は要約を生成して追加
            if metadata.get("content"):
                # content_summaryという名前で本文の要約を保存
                metadata["content_summary"] = generate_summary(metadata["content"], client)
                print(f"  本文要約を生成しました ({len(metadata['content_summary'])}文字)")

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

        # 図グループの要約を生成
        print("\n図グループの要約を生成中...")
        grouped_summaries = {}  # {ベース図ID: 要約} の辞書

        for file_path, metadata in all_metadata.items():
            if metadata.get("grouped_figure_contexts"):
                for base_id, group_data in metadata["grouped_figure_contexts"].items():
                    if base_id not in grouped_summaries:
                        summary = generate_figure_group_summary(base_id, group_data, client)
                        if summary:
                            grouped_summaries[base_id] = summary
                            included_figs = ", ".join(group_data["included_ids"])
                            print(f"  {base_id}グループ({included_figs})の要約を生成しました ({len(summary)}文字)")

        # ユーザー確認 - embeddingを行うかどうか
        proceed = input("\nembedding計算を開始しPineconeにデータを送信しますか？ (y/n): ").lower()
        if proceed != 'y':
            print("処理を中止します")
            return
        
        # ここに変数の宣言を移動（すべてのembedding処理の前に配置）
        vectors_to_upsert = []
        text_count = 0
        image_count = 0
        content_count = 0

        # 本文の独立したエンベディングを生成
        print("\n本文のエンベディングを生成中...")
        content_count = 0
        for file_path, metadata in all_metadata.items():
            if metadata.get("content"):
                # ファイル名から一意のIDを生成
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                content_id = f"{base_name}_content"
                
                # 本文のエンベディングを生成
                content_emb = get_text_embedding(metadata["content"], client)
                content_count += 1
                
                # 本文のメタデータを設定
                content_metadata = {
                "category": "dental",
                "data_type": "content",
                "title": metadata.get("title", ""),
                "item": metadata.get("item", ""),
                "content_summary": metadata.get("content_summary", ""),
                # 本文自体も含める（長すぎる場合は適切に切り詰める）
                "content_text": metadata.get("content", "")[:2000]  # 最初の2000文字だけ保存
            }
                
                # 関連する図表IDを追加
                related_figs = [k for k, v in all_entries.items() if k.startswith(base_name)]
                if related_figs:
                    content_metadata["related_figures"] = related_figs
                
                vectors_to_upsert.append((
                    content_id,
                    content_emb,
                    content_metadata
                ))
                
                print(f"  {content_id}の本文エンベディングを生成しました")

        # y/n確認後にembedding計算を開始
        # print("\n図表のembedding計算を開始します...")
        # vectors_to_upsert = []
        # text_count = 0
        # image_count = 0

        for item in entries_to_process:
            base_id = item["base_id"]
            data = item["text_data"]
            image_path = item["image_path"]
            text_file_part = item["text_file_part"]
            entry_type = item["entry_type"]
            base_num = item["base_num"]
            variation = item["variation"]
            
            # メタデータを含むテキスト情報の作成
            metadata_info = data.get("metadata", {})
            
            # 図のベースIDを取得（例: Fig1a -> Fig1）
            fig_base_id = f"{entry_type}{base_num}"
            
            # メタデータテキストの構築
            metadata_text = ""
            if metadata_info.get("title"):
                metadata_text += f"title: {metadata_info['title']}\n"                        
            if metadata_info.get("item"):
                metadata_text += f"item: {metadata_info['item']}\n"
            if metadata_info.get("content_summary"):
                metadata_text += f"content_summary: {metadata_info['content_summary']}\n"
            
            # グループ要約があれば追加
            if fig_base_id in grouped_summaries:
                metadata_text += f"図グループ要約: {grouped_summaries[fig_base_id]}\n"
            
            # メタデータとオリジナルテキストを組み合わせる
            combined_text = metadata_text + data["text"] if metadata_text else data["text"]
            
            # テキストのembedding取得
            print(f"テキストのembedding計算中: {data['text_id']}")
            text_emb = get_text_embedding(combined_text, client)
            text_count += 1
            
            # メタデータをメタデータフィールドに追加
            metadata_fields = {
                "category": "dental",
                "data_type": "text",
                "text": data["text"],
                "entry_type": entry_type,  # "Fig" または "case"
                "related_image_id": data["image_id"]
            }
            
            # メタデータ情報をメタデータフィールドに追加
            if metadata_info.get("title"):
                metadata_fields["title"] = metadata_info["title"]                       
            if metadata_info.get("item"):
                metadata_fields["item"] = metadata_info["item"]
            # 本文は含めない（容量削減のため）
            # if metadata_info.get("content"):
            #     metadata_fields["content"] = metadata_info["content"][:500]
            # 代わりに本文の要約を含める
            if metadata_info.get("content_summary"):
                metadata_fields["content_summary"] = metadata_info["content_summary"]
            
            # 図グループ要約があれば追加
            if fig_base_id in grouped_summaries:
                metadata_fields["figure_group_summary"] = grouped_summaries[fig_base_id]
                metadata_fields["figure_group_id"] = fig_base_id
            
            # 関連する本文エントリーIDを追加
            content_id = f"{text_file_part}_content"
            metadata_fields["related_content_id"] = content_id
            
            vectors_to_upsert.append((
                data["text_id"],
                text_emb,
                metadata_fields
            ))
            
            # 画像のembedding取得
            print(f"画像のembedding計算中: {image_path}")
            image_result = get_image_embedding(image_path, model, preprocess, device)
            if image_result["status"] == "success":
                image_count += 1
                image_emb = image_result["vector"]
                image_id = f"{text_file_part}_{entry_type}{base_num}{variation}_image"
                
                # 画像のメタデータフィールドも同様に設定
                image_metadata_fields = {
                    "category": "dental",
                    "data_type": "image",
                    "text": data["text"],
                    "entry_type": entry_type,  # "Fig" または "case"
                    "related_text_id": data["text_id"]
                }
                
                # メタデータ情報を画像のメタデータフィールドにも追加
                if metadata_info.get("title"):
                    image_metadata_fields["title"] = metadata_info["title"]                               
                if metadata_info.get("item"):
                    image_metadata_fields["item"] = metadata_info["item"]
                # 本文は含めない（容量削減のため）
                # if metadata_info.get("content"):
                #     image_metadata_fields["content"] = metadata_info["content"][:500]
                # 代わりに本文の要約を含める
                if metadata_info.get("content_summary"):
                    image_metadata_fields["content_summary"] = metadata_info["content_summary"]
                
                # 図グループ要約があれば画像メタデータにも追加
                if fig_base_id in grouped_summaries:
                    image_metadata_fields["figure_group_summary"] = grouped_summaries[fig_base_id]
                    image_metadata_fields["figure_group_id"] = fig_base_id
                
                # 関連する本文エントリーIDを追加
                image_metadata_fields["related_content_id"] = content_id
                
                vectors_to_upsert.append((
                    image_id,
                    image_emb,
                    image_metadata_fields
                ))
            
            print(f"- {base_id}の処理完了")

        print(f"\n処理したテキスト数: {text_count}")
        print(f"処理した画像数: {image_count}")
        print(f"処理した本文数: {content_count}")

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