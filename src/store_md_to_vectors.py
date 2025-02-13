import chromadb
import uuid
from pathlib import Path
import json
from langchain.text_splitter import MarkdownHeaderTextSplitter

# def chunk_text(text, max_tokens=512, encoding="cl100k_base"):
#     tokenizer = tiktoken.get_encoding(encoding)
#     tokens = tokenizer.encode(text)
    
#     # 最大トークン数で分割
#     for i in range(0, len(tokens), max_tokens):
#         yield tokenizer.decode(tokens[i:i + max_tokens])

# **埋め込みモデルをロード**
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        return_each_line=True
    )

def store_md_to_vector_db(db_path: str, segment_folder: str):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="md_details")

    for segment_file in Path(segment_folder).glob("*.json"):

        print(f"`----------------{segment_folder}/{segment_file}の処理を開始します----------------")

        with open(segment_file, "r", encoding="utf-8") as file:
            metadata_json = json.load(file) 

        text = segment_file.read_text(encoding="utf-8")  # Markdownファイルを文字列として取得
        texts = markdown_splitter.split_text(text)
        for text in texts:
            print(text)
        
            # 一意の ID を生成して追加
            unique_id = str(uuid.uuid4())

            metadata_str = str(text.metadata)  # metadata を文字列に変換

            collection.add(
                ids=[unique_id],  # 必須の ID を追加
                documents=text.page_content,
                # embeddings=[embedding_vector],  # **埋め込みベクトルを追加**
                metadatas=[
                    {
                        "company": metadata_json["company"],
                        "title": metadata_json["title"], 
                        "header": metadata_str, 
                        "contents": metadata_json["contents"],
                        "filename": metadata_json["filename"], 
                        "page": metadata_json["page"]
                    }
                ]
            )

    for segment_file in Path(segment_folder).glob("*.md"):

        print(f"`----------------{segment_folder}/{segment_file}の処理を開始します----------------")

        # メタデータの取得
        metadata_path = list(Path(segment_folder).glob(f"{segment_file.stem}.json"))
        metadata_path = metadata_path[0]
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata_json = json.load(file) 

        text = segment_file.read_text(encoding="utf-8")  # Markdownファイルを文字列として取得
        texts = markdown_splitter.split_text(text)
        for text in texts:
            print(text)
        
            # 一意の ID を生成して追加
            unique_id = str(uuid.uuid4())

            metadata_str = str(text.metadata)  # metadata を文字列に変換

            collection.add(
                ids=[unique_id],  # 必須の ID を追加
                documents=text.page_content,
                # embeddings=[embedding_vector],  # **埋め込みベクトルを追加**
                metadatas=[
                    {
                        "company": metadata_json["company"],
                        "title": metadata_json["title"], 
                        "header": metadata_str, 
                        "contents": metadata_json["contents"],
                        "filename": metadata_json["filename"], 
                        "page": metadata_json["page"]
                    }
                ]
            )

# メタデータをChromaDBに格納
def metadata_to_vector_db(db_path: str, segment_folder: str):

    # 会社名を格納するDB
    client_companyname = chromadb.PersistentClient(path=db_path)
    collection_companyname = client_companyname.get_or_create_collection(name="company_name")

    # PDF記載内容の概要を格納するDB
    client_contents = chromadb.PersistentClient(path=db_path)
    collection_contents = client_contents.get_or_create_collection(name="contents")

    for segment_file in Path(segment_folder).glob("*.json"):

        print(f"`----------------{segment_folder}/{segment_file}の処理を開始します----------------")

        with open(segment_file, "r", encoding="utf-8") as file:
            metadata_json = json.load(file) 
        
        # 一意の ID を生成して追加
        unique_id = str(uuid.uuid4())

        # 会社名をChromaDBに格納
        collection_companyname.add(
            ids=[unique_id],  # 必須の ID を追加
            documents=metadata_json["company"],
            metadatas=[
                {
                    "company": metadata_json["company"],
                    "title": metadata_json["title"], 
                    "filename": metadata_json["filename"], 
                    "page": metadata_json["page"]
                }
            ]
        )

        # 一意の ID を生成して追加
        unique_id = str(uuid.uuid4())

        collection_contents.add(
            ids=[unique_id],  # 必須の ID を追加
            documents=metadata_json["contents"],
            metadatas=[
                {
                    "company": metadata_json["company"],
                    "title": metadata_json["title"], 
                    "filename": metadata_json["filename"], 
                    "page": metadata_json["page"]
                }
            ]
        )


if __name__ == "__main__":
    store_md_to_vector_db("data/vector_db", "data/vector_db")
