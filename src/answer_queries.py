import os
import chromadb
from dotenv import load_dotenv
from os.path import join, dirname
from openai import AzureOpenAI
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

def search_vector_db(db_path: str, query: str):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="reports")
    # results = collection.query(query_texts=[query], n_results=10)
    # return [doc for doc in results["documents"][0]]

    # **埋め込みモデルをロード**
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

     # クエリをベクトルに変換
    query_vector = embedding_model.encode(query).tolist()

    # ベクトル検索を実行
    results = collection.query(query_embeddings=[query_vector], n_results=10)

    # 結果が空の場合の処理
    if not results["documents"]:
        return []

    # # 関連するドキュメントとメタデータを取得
    # return [
    #     {"document": doc, "metadata": meta}
    #     for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    # ]
    return [doc for doc in results["documents"][0]]   

def answer_questions(query_file: str, db_path: str, output_file: str):

    # 環境変数読み込み
    load_dotenv(verbose=True)

    client = AzureOpenAI(
        api_key = os.getenv('AZURE_OPENAI_API_KEY'),  
        api_version = "2024-10-21",
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        )

    queries = pd.read_csv(query_file)

    answers = []
    for _, row in queries.iterrows():
        relevant_docs = search_vector_db(db_path, row["problem"])
        context = " ".join(relevant_docs)

        role = "あなたは企業に関する質問に対して、情報を必要に応じてベクトルデータベースから検索し、回答するRAGです。"
        
        prompt = f"""

            次の質問に54トークン以内で簡潔に回答してください。: {row['problem']}\n
            あなたがベクトルデーターベースから質問に関連して検索した情報を示します。こちらを参考に回答してください。Context: {context}\n
            回答の際には、以下に注意して回答してください。
            # 文章ではなく、単語レベルで完結に回答すること。
            # 回答が複数ある場合には、句点(、)で区切って回答をすること。
            # 数値を答えさせる問題の場合は、数値のみを回答すること。
            """

        response = client.chat.completions.create(
            model="4omini",
            temperature=0,
            # max_tokens =54,
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content.strip()
        if len(answer.split()) > 54:
            answer = " ".join(answer.split()[:54])  # 54トークンに制限
        if answer == "":
            answer = "分かりません"

        #####
        num = row["index"]
        print(f"{num}番目の問題の回答：{answer}")
        #####

        answers.append({"index": row["index"], "answer": answer})

    pd.DataFrame(answers).to_csv(output_file, index=False)


def answer_questions_second(query_file: str, db_path: str, output_file: str, answer_file: str):

    # 環境変数読み込み
    load_dotenv(verbose=True)

    client = AzureOpenAI(
        api_key = os.getenv('AZURE_OPENAI_API_KEY'),  
        api_version = "2024-10-21",
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        )

    queries = pd.read_csv(query_file)
    answer_first = pd.read_csv(answer_file)

    answers = []
    for _, row in queries.iterrows():
        relevant_docs = search_vector_db(db_path, row["problem"])
        context = " ".join(relevant_docs)
        index = int(row["index"])
        primary_answer = answer_first[answer_first["index"] == index]

        role = "あなたは企業に関する質問に対して、情報を必要に応じてベクトルデータベースから検索し、回答するRAGです。"
        
        prompt = f"""

            次の質問に54トークン以内で簡潔に回答してください。: {row['problem']}\n
            あなたがベクトルデーターベースから質問に関連して検索した情報を示します。こちらを参考に回答してください。Context: {context}\n
            また、あなたの質問に対する一次回答を示します。こちらを参考に一次回答で回答分からなかった問題について再度考えてください。一次回答：{primary_answer}\n
            一次回答で回答できた問題に関しては、一次回答での結果をそのまま答えてください。
            再度考えた結果、それでもわからない場合は「分かりません」とのみ回答してください。
            回答の際には、以下に注意して回答してください。
            # 文章ではなく、単語レベルで完結に回答すること。
            # 回答が複数ある場合には、句点(、)で区切って回答をすること。
            # 数値を答えさせる問題の場合は、数値のみを回答すること。
            """

        response = client.chat.completions.create(
            model="4omini",
            temperature=0,
            # max_tokens =54,
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content.strip()
        if len(answer.split()) > 54:
            answer = " ".join(answer.split()[:54])  # 54トークンに制限
        if answer == "":
            answer = "分かりません"

        #####
        num = row["index"]
        print(f"{num}番目の問題の回答：{answer}")
        #####

        answers.append({"index": row["index"], "answer": answer})

    pd.DataFrame(answers).to_csv(output_file, index=False)

if __name__ == "__main__":
    answer_questions("data/query.csv", "data/vector_db", "data/prediction.csv")
