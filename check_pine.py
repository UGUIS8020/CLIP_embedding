import os
from pinecone import Pinecone
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import json

def initialize_pinecone():
    """Pineconeの初期化を行う"""
    try:
        load_dotenv()
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY が設定されていません")
        
        pc = Pinecone(api_key=pinecone_api_key)
        return pc.Index("raiden")
    except Exception as e:
        rprint(f"[red]Pinecone の初期化でエラーが発生しました: {e}[/red]")
        raise

def format_metadata(metadata):
    """メタデータを見やすく整形する"""
    formatted = {}
    for key, value in metadata.items():
        if isinstance(value, str) and len(value) > 100:
            # テキストが長い場合は省略
            formatted[key] = value[:100] + "..."
        else:
            formatted[key] = value
    return formatted

def display_vector_info(vector_id, metadata, console):
    """ベクトル情報を表示する"""
    rprint(f"\n[bold blue]📎 ベクトル ID: {vector_id}[/bold blue]")
    
    # メタデータをテーブルで表示
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("キー", style="dim")
    table.add_column("値", style="yellow")
    
    formatted_metadata = format_metadata(metadata)
    for key, value in formatted_metadata.items():
        # 長いテキストは改行して表示
        if isinstance(value, str):
            value = value.replace('\n', ' ')
        table.add_row(key, str(value))
    
    console.print(table)

def main():
    """メイン処理"""
    try:
        console = Console()
        index = initialize_pinecone()
        
        # 統計情報の取得
        stats = index.describe_index_stats()
        rprint(f"\n[bold green]📊 保存されているベクトル数: {stats['total_vector_count']}[/bold green]")
        
        # データの取得
        rprint("\n[bold]🔍 登録データを取得中...[/bold]")
        query_result = index.query(
            vector=[0.0] * 1536,
            top_k=50,
            include_metadata=True
        )
        
        if not query_result or "matches" not in query_result:
            rprint("[red]⚠️ データが見つかりませんでした[/red]")
            return
        
        # データの表示
        rprint(f"\n[bold green]✨ 登録データ一覧（最大50件）:[/bold green]")
        for match in query_result["matches"]:
            display_vector_info(match['id'], match.get('metadata', {}), console)
        
        rprint("\n[bold green]✅ データ確認完了[/bold green]")
        
    except Exception as e:
        rprint(f"[red]エラーが発生しました: {e}[/red]")
        return

if __name__ == "__main__":
    main()