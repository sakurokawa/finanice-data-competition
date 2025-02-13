import os
import chromadb
from dotenv import load_dotenv
from os.path import join, dirname
from openai import AzureOpenAI
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import json

def search_vector_db(db_path: str, query: str, question_sum:dict):
    client_details = chromadb.PersistentClient(path=db_path)
    collection_details = client_details.get_collection(name="md_details")

    client_companyname = chromadb.PersistentClient(path=db_path)
    collection_companyname = client_companyname.get_collection(name="company_name")

    client_contents = chromadb.PersistentClient(path=db_path)
    collection_contents = client_contents.get_collection(name="contents")


    # **埋め込みモデルをロード**
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 各要素のベクトルを取得
    query_vector = embedding_model.encode(query).tolist()
    company_vector = embedding_model.encode(question_sum["company"]).tolist()
    topic_vector = embedding_model.encode(question_sum["info"]).tolist()

    # 各要素の重み（調整可能）
    alpha = 0.1  # `query` の重み
    beta = 0.8   # `company` の重み
    gamma = 0.1  # `topic` の重み

    # ベクトルの加重平均を計算
    combined_vector = [
        (alpha * q + beta * c + gamma * t) 
        for q, c, t in zip(query_vector, company_vector, topic_vector)
    ]

    # 会社名について、ベクトル検索を実行
    

    # ベクトル検索を実行
    results = collection.query(
        query_embeddings=[combined_vector],
        n_results=5
    )

    # 結果が空の場合の処理
    if not results["documents"]:
        return []

    # 関連するドキュメントとメタデータを取得
    # return [
    #     {"document": doc, "metadata": meta}
    #     for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    # ]
    return [
        {"company":meta["company"], "contents": meta["contents"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

def question_summary(problem: str):

    # 環境変数読み込み
    load_dotenv(verbose=True)

    client = AzureOpenAI(
        api_key = os.getenv('AZURE_OPENAI_API_KEY'),  
        api_version = "2024-10-21",
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        )

    role = "あなたは企業に関する質問に対して、情報を必要に応じてベクトルデータベースから検索し、回答するRAGです。"
    
    prompt = f"""

        次の質問文は、ある会社の統合報告書あるいはそれに類するレポートに記載のある情報に関する質問文です。
        質問に関する情報を次の指示に従って漏れがないようにJSON形式で出力してください。\n
        【質問文】: {problem}\n
        1.質問文で言及されている企業について、JSON内のcompanyに出力してください。\n
        2.質問文にに対して回答するために必要な情報を簡潔に表現し、JSON内のinfoに出力してください。ここでは質問に対する直接の回答を記載する必要はありません。\n
        3.infoは全体で50文字程度で簡潔に出力してください。回答をするために必要な情報が複数存在する場合には、改行は使用せず、句点(、)で区切って出力してください。\n
        4.JSONによる出力は、以下に従ってください。【例】: {{"company":"株式会社ABC","info":"2024年度の経理部の新入社員入社人数に関する情報"}}\n
        5.日本語で出力してください。\n
        6.出力にあなたの言葉は含めず、質問文から得られた情報のみを使って出力してください。\n
        """

    response = client.chat.completions.create(
        model="4omini",
        temperature=0,
        response_format={ "type": "json_object" },
        # max_tokens =54,
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content.strip()
    response_json = json.loads(answer)
    print(response_json)

    return response_json

def answer_questions_md(query_file: str, db_path: str, output_file: str):

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
        question_sum = question_summary(row["problem"])
        relevant_docs = search_vector_db(db_path, row["problem"],question_sum)

        print(f"問題：{row["problem"]}")
        print(f"関連箇所：{relevant_docs}")

        role = "あなたは企業に関する質問に対して、情報を必要に応じてベクトルデータベースから検索し、回答するRAGです。"
        
        prompt = f"""

            次の質問に54トークン以内で簡潔に回答してください。: {row['problem']}\n
            あなたがベクトルデーターベースから質問に関連して検索した情報を示します。こちらを参考に回答してください。\n
            【参考文献】\n
            {relevant_docs}
            回答の際には、以下に注意して回答してください。\n
            # 回答には必ず参考文献の情報のみを用い、あなたの言葉では回答しないこと。
            # 質問がどの会社についての質問かをまず判定し、参考とすべき情報を判断すること。\n
            # 文章ではなく、単語レベルで完結に回答すること。\n
            # 回答が複数ある場合には、句点(、)で区切って回答をすること。\n
            # 数値を答えさせる問題の場合は、数値のみを回答すること。\n
            # 回答できない場合は、「分かりません。」と述べ、その後に回答するために不足している情報を述べてください。
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


def answer_questions_md_second(query_file: str, db_path: str, output_file: str, answer_file: str):

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
        question_sum = question_summary(row["problem"])
        relevant_docs = search_vector_db(db_path, row["problem"],question_sum)
        # context = " ".join(relevant_docs)
        index = int(row["index"])
        primary_answer = answer_first[answer_first["index"] == index]

        role = "あなたは企業に関する質問に対して、情報を必要に応じてベクトルデータベースから検索し、回答するRAGです。"
        
        prompt = f"""

            次の質問に54トークン以内で簡潔に回答してください。: {row['problem']}\n
            あなたがベクトルデーターベースから質問に関連して検索した情報を示します。こちらを参考に回答してください。\n
            【参考文献】\n
            {relevant_docs}\n
            また、あなたの質問に対する一次回答を示します。こちらを参考に一次回答で回答分からなかった問題について再度考えてください。一次回答：{primary_answer}\n
            一次回答で回答できた問題に関しては、一次回答での結果をそのまま答えてください。\n
            再度考えた結果、それでもわからない場合は「分かりません」とのみ回答してください。\n
            回答の際には、以下に注意して回答してください。\n
            # 文章ではなく、単語レベルで完結に回答すること。\n
            # 回答が複数ある場合には、句点(、)で区切って回答をすること。\n
            # 数値を答えさせる問題の場合は、数値のみを回答すること。\n
            """

        response = client.chat.completions.create(
            model="4omini",
            temperature=0,
            max_tokens =54,
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
    answer_questions_md("data/query.csv", "data/vector_db", "data/prediction.csv")
