import os
from pathlib import Path
from datetime import datetime

class TextProcessor:
    def __init__(self):
        self.text_dir = Path("text")
        self.output_dir = Path("data")
        self.backup_dir = Path("backup")
        
        self.setup_directories()

    def setup_directories(self):
        """必要なディレクトリ構造を作成"""
        dirs = [self.text_dir, self.output_dir, self.backup_dir]
        for dir_path in dirs:
            dir_path.mkdir(exist_ok=True)

    def normalize_text(self, text):
        """テキストの正規化処理"""
        replacements = {
            # 数字の正規化
            '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
            '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
            
            # アルファベット大文字の正規化
            'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E',
            'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J',
            'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O',
            'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T',
            'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X', 'Ｙ': 'Y',
            'Ｚ': 'Z',
            
            # アルファベット小文字の正規化
            'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e',
            'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j',
            'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o',
            'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's', 'ｔ': 't',
            'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x', 'ｙ': 'y',
            'ｚ': 'z',
            
            # 括弧の正規化
            '（': '(', '）': ')', '［': '[', '］': ']',
            '｛': '{', '｝': '}', '【': '[', '】': ']',
            '「': '"', '」': '"', '『': '"', '』': '"',
            
            # 句読点の正規化
            '、': ',', '，': ',', '。': '.', '．': '.',
            
            # その他の記号
            '：': ':', '；': ';', '！': '!', '？': '?',
            '＝': '=', '％': '%', '＋': '+', '－': '-',
            '＊': '*', '／': '/', '　': ' ',  # 全角スペースを半角に
            '＆': '&', '＠': '@', '＾': '^', '＿': '_',
            '｜': '|', '＜': '<', '＞': '>', '～': '~'
        }
        
        normalized_text = text
        for old, new in replacements.items():
            normalized_text = normalized_text.replace(old, new)
        
        return normalized_text

    def create_backup(self, file_path):
        """ファイルのバックアップを作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')
        return backup_path

    def process_files(self):
        """textフォルダ内のすべてのテキストファイルを処理"""
        if not any(self.text_dir.iterdir()):
            print("Warning: No files found in text directory.")
            return

        for file_path in self.text_dir.glob("*.txt"):
            try:
                print(f"Processing {file_path.name}...")
                
                # バックアップを作成
                backup_path = self.create_backup(file_path)
                print(f"Backup created: {backup_path.name}")
                
                # ファイルを読み込んで処理
                text = file_path.read_text(encoding='utf-8')
                normalized_text = self.normalize_text(text)
                
                # 処理結果を出力
                output_path = self.output_dir / f"{file_path.name}"
                output_path.write_text(normalized_text, encoding='utf-8')
                
                print(f"Processed file saved: {output_path.name}")
                
                # 変更の分析
                self.analyze_changes(text, normalized_text)
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")

    def analyze_changes(self, original_text, normalized_text):
        """変更内容の分析と表示"""
        changes = {}
        for orig, norm in zip(original_text, normalized_text):
            if orig != norm:
                key = f"{orig} -> {norm}"
                changes[key] = changes.get(key, 0) + 1
        
        if changes:
            print("\nChanges made:")
            for change, count in changes.items():
                print(f"{change}: {count} occurrences")
        else:
            print("\nNo changes were necessary")

def main():
    processor = TextProcessor()
    print("Starting text processing...")
    processor.process_files()
    print("\nProcessing completed.")

if __name__ == "__main__":
    main()