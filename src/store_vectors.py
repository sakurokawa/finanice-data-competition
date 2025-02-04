import chromadb
import json
import uuid
from pathlib import Path
# import tiktoken
from sentence_transformers import SentenceTransformer

# def chunk_text(text, max_tokens=512, encoding="cl100k_base"):
#     tokenizer = tiktoken.get_encoding(encoding)
#     tokens = tokenizer.encode(text)
    
#     # 最大トークン数で分割
#     for i in range(0, len(tokens), max_tokens):
#         yield tokenizer.decode(tokens[i:i + max_tokens])

# **埋め込みモデルをロード**
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def store_segments_to_vector_db(db_path: str, segment_folder: str):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="reports")

    print(f"`----------------{segment_folder}の処理を開始します----------------")

    for segment_file in Path(db_path/segment_folder).glob("*.json"):
        with open(segment_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: {segment_file} の JSON が不正です。スキップします。")
                continue

        # # `data` はリスト形式であることを確認
        # if not isinstance(data, list):
        #     print(f"Warning: {segment_file} のデータ形式が不正です。スキップします。")
        #     continue
        
        # `company` と`info` の存在を確認
        if not "company" not in data or "info" not in data:
            print(f"Warning: {segment_file} のセグメントデータが不正です。スキップします。")
            continue

        # 企業名を取り出し
        company_name = data["company"]

        # for segment in data:
        #     # `company` と`topic` と `details` の存在を確認
        #     if not isinstance(segment, dict) or "company" not in segment or "topic" not in segment or "details" not in segment:
        #         print(f"Warning: {segment_file} のセグメントデータが不正です。スキップします。")
        #         continue

        for details in data["info"]:
            # `topic` と`details` の存在を確認
            if not "topic" not in details or "details" not in details:
                print(f"Warning: {segment_file} のセグメントデータが不正です。スキップします。")
                continue

            # トピック名を取り出し
            topic_name = data["topic"]
            
            # `details` 内の各テキストをデータベースに追加
            for detail in details["details"]:
                if not isinstance(detail, str):
                    print(f"Warning: {segment_file} の 'details' 内に無効なデータがあります。スキップします。")
                    continue

                # **テキストを埋め込みベクトルに変換**
                embedding_vector = embedding_model.encode(detail).tolist()

                # 一意の ID を生成して追加
                unique_id = str(uuid.uuid4())

                collection.add(
                    ids=[unique_id],  # 必須の ID を追加
                    documents=[detail],
                    embeddings=[embedding_vector],  # **埋め込みベクトルを追加**
                    metadatas=[{"company": company_name, "topic": topic_name, "source": segment_file.stem}]
                )

                # # チャンクごとにデータベースに追加
                # for chunk in chunk_text(detail):
                #     unique_id = str(uuid.uuid4())
                #     collection.add(
                #         ids=[unique_id],
                #         documents=[chunk],
                #         metadatas=[{"company": segment["company"], "topic": segment["topic"], "source": segment_file.stem}]
                #     )

if __name__ == "__main__":
    store_segments_to_vector_db("data/vector_db", "data/vector_db")
