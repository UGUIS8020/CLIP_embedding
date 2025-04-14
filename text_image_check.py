import os
import re
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# カテゴリー設定
CATEGORY = "dental"

@dataclass
class Metadata:
    title: str = ""
    content: str = ""
    figure_descriptions: Dict[str, str] = field(default_factory=dict)
    case_descriptions: Dict[str, str] = field(default_factory=dict)
    topic: str = ""

@dataclass
class Entry:
    type: str
    number: str
    text: str
    text_id: str
    image_id: str
    metadata: Metadata
    base_name: str = ""

class TextProcessor:
    def __init__(self):
        # 正規表現パターン
        self.figure_pattern = re.compile(r'\[(Fig\d+(?:[a-z]*))(?:\]|$)')
        self.case_pattern = re.compile(r'\[(case\d+(?:[a-z]*))(?:\]|$)')
        self.any_pattern = re.compile(r'\[((Fig|case)\d+(?:[a-z]*))(?:\]|$)')

    def process_file(self, file_path: str) -> List[Entry]:
        """テキストファイルを処理し、メタデータと図・ケースの説明を抽出"""
        entries = []
        try:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            metadata = Metadata()
            
            # ファイル名からメタデータを抽出
            filename = os.path.basename(file_path)
            metadata.title = filename.split('.')[0]
            
            # その後、title[]から抽出
            lines = text.split('\n')
            for line in lines:
                title_match = re.search(r'title\[(.*?)\]', line)
                if title_match:
                    metadata.title = title_match.group(1).strip()
            
            # 本文と図・ケースの説明を分離
            main_content, illustration_content = self._split_content_and_illustrations(text)
            metadata.content = main_content.strip()
            
            # 図・ケースの説明を処理
            if illustration_content:
                self._process_illustration_descriptions(illustration_content, metadata)
            
            # topicの抽出
            topic_match = re.search(r'topic\[(.*?)\]', text)
            if topic_match:
                metadata.topic = topic_match.group(1).strip()
            
            # Entryを作成する際にbase_nameを設定
            entry = Entry(
                type="content",
                number="",
                text=metadata.content,
                text_id=file_path,
                image_id="",
                metadata=metadata,
                base_name=base_name
            )
            entries.append(entry)
            
            # 図の説明文の処理
            for fig_id, description in metadata.figure_descriptions.items():
                if description:
                    entry = Entry(
                        type="figure_description",
                        number=fig_id,
                        text=description,
                        text_id=f"{base_name}_{fig_id}_desc",
                        image_id=f"{base_name}_{fig_id}_image",
                        metadata=metadata,
                        base_name=base_name
                    )
                    entries.append(entry)
                    print(f"図の説明エントリー追加: {fig_id} -> {entry.text_id}")
            
            # ケースの説明文の処理
            for case_id, description in metadata.case_descriptions.items():
                if description:
                    entry = Entry(
                        type="case_description",
                        number=case_id,
                        text=description,
                        text_id=f"{base_name}_{case_id}_desc",
                        image_id=f"{base_name}_{case_id}_image",
                        metadata=metadata,
                        base_name=base_name
                    )
                    entries.append(entry)
                    print(f"ケースの説明エントリー追加: {case_id} -> {entry.text_id}")
            
        except Exception as e:
            print(f"テキスト抽出エラー: {e}")
        
        return entries

    def _split_content_and_illustrations(self, text: str) -> Tuple[str, str]:
        """本文と図・ケースの説明を分離"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            # Fig または case のいずれかのパターンにマッチするか確認
            if self.any_pattern.match(line.strip()):
                return '\n'.join(lines[:i]).strip(), '\n'.join(lines[i:])
        return text.strip(), ""

    def _process_illustration_descriptions(self, illustration_content: str, metadata: Metadata):
        """図・ケースの説明文を処理"""
        current_id = None
        current_text = []
        current_type = None  # "figure" または "case"
        
        for line in illustration_content.split('\n'):
            # Figパターンのチェック
            fig_match = self.figure_pattern.match(line.strip())
            # caseパターンのチェック
            case_match = self.case_pattern.match(line.strip())
            
            if fig_match:
                # 前のエントリーを保存
                if current_id and current_text:
                    if current_type == "figure":
                        metadata.figure_descriptions[current_id] = '\n'.join(current_text).strip()
                    elif current_type == "case":
                        metadata.case_descriptions[current_id] = '\n'.join(current_text).strip()
                
                current_id = fig_match.group(1)
                current_text = []
                current_type = "figure"
                print(f"図説明を検出: {current_id}")
            elif case_match:
                # 前のエントリーを保存
                if current_id and current_text:
                    if current_type == "figure":
                        metadata.figure_descriptions[current_id] = '\n'.join(current_text).strip()
                    elif current_type == "case":
                        metadata.case_descriptions[current_id] = '\n'.join(current_text).strip()
                
                current_id = case_match.group(1)
                current_text = []
                current_type = "case"
                print(f"ケース説明を検出: {current_id}")
            elif line.strip() and current_id:
                current_text.append(line.strip())
        
        # 最後のエントリーを保存
        if current_id and current_text:
            if current_type == "figure":
                metadata.figure_descriptions[current_id] = '\n'.join(current_text).strip()
            elif current_type == "case":
                metadata.case_descriptions[current_id] = '\n'.join(current_text).strip()

def main():
    try:
        # プロセッサーの初期化
        text_processor = TextProcessor()
        
        # テキストファイルの処理
        txt_files = glob.glob("data/*.txt")
        if not txt_files:
            raise FileNotFoundError("data ディレクトリにtxtファイルが見つかりません")

        # 本文と画像説明の抽出
        all_entries = []
        for file_path in txt_files:
            print(f"\nテキストファイル処理: {file_path}")
            entries = text_processor.process_file(file_path)
            all_entries.extend(entries)

        # 画像ファイルの確認
        image_files = glob.glob(os.path.join("data", "Fig*.jpg"))
        case_image_files = glob.glob(os.path.join("data", "case*.jpg"))
        
        # デバッグ: 抽出されたすべての図とケースを表示
        print("\n== 抽出された図の説明 ==")
        for entry in all_entries:
            if entry.type == "figure_description":
                print(f"ID: {entry.number}, Text ID: {entry.text_id}, Image ID: {entry.image_id}")
                
        print("\n== 抽出されたケースの説明 ==")
        for entry in all_entries:
            if entry.type == "case_description":
                print(f"ID: {entry.number}, Text ID: {entry.text_id}, Image ID: {entry.image_id}")

        # 抽出結果の表示と確認
        print("\n=== 抽出結果の確認 ===")
        content_count = sum(1 for e in all_entries if e.type == "content")
        fig_desc_count = sum(1 for e in all_entries if e.type == "figure_description")
        case_desc_count = sum(1 for e in all_entries if e.type == "case_description")
        print(f"メインテキスト数: {content_count}")
        print(f"画像説明テキスト数: {fig_desc_count}")
        print(f"ケース説明テキスト数: {case_desc_count}")
        print(f"Fig画像ファイル数: {len(image_files)}")
        print(f"ケース画像ファイル数: {len(case_image_files)}")

        print("\n画像テキストと画像ファイルの対応:")
        print("\n図テキスト:")
        for entry in all_entries:
            if entry.type == "figure_description":
                print(f"- {entry.text_id} -> {entry.image_id}")
        
        print("\nケーステキスト:")
        for entry in all_entries:
            if entry.type == "case_description":
                print(f"- {entry.text_id} -> {entry.image_id}")

        print("\nFig画像ファイル:")
        for img in sorted(image_files):
            print(f"- {os.path.basename(img)}")
            
        print("\nケース画像ファイル:")
        for img in sorted(case_image_files):
            print(f"- {os.path.basename(img)}")

    except Exception as e:
        print(f"\n処理中にエラーが発生: {e}")

if __name__ == "__main__":
    main()