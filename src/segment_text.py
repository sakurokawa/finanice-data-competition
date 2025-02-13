import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import typing
from pathlib import Path
from PIL import Image
import io
import google.api_core.exceptions
from google.api_core import retry
from utils.schema import Discription
from pathlib import Path

@retry.Retry()
def segment_image_with_gemini(pdf_folder: str):
    
    #pathの末尾を取得(ex.1_pdf)
    # pdf_folder_end = pdf_folder.removeprefix("data/images/")
    pdf_folder_end = pdf_folder.removeprefix("validation/images/")


    # 環境変数読み込み
    load_dotenv(verbose=True)
    genai.configure(api_key = os.getenv('GEMINI_API_KEY '))
    model = genai.GenerativeModel("gemini-1.5-flash")

    print(f"{pdf_folder}について処理を開始します")

    prompt = """
        
        次の画像は、ある会社の統合報告書あるいはそれに類するレポートの一部です。\n
        画像にある内容を次の指示に従って漏れがないようにJSOON形式で出力してください。\n
        1.複数のトピック(topic)と詳細(detail)に分けて返却してください。
        2.文章や表、画像や内容によってトピック(topic)に分割し、レイアウトを踏まえてトピック(topic)ごとに漏れなく内容を詳細(detail)に返却してください。
        3. 複数のトピック(topic)と詳細(detail)はinfoとしてまとめてください。
        4.また、画像で言及されている情報がどの会社の情報かを特定し、JSON内のcompanyに設定してください。
        5.トピックが文章を含んでいる場合は、文章は漏れなく出力してください。\n
        6.トピックが表を含んでいる場合には、文章は漏れなく出力してください。\n
        7.トピックが図を含んでいる場合には、代わりに文章で図の内容を表現してください。\n
        8.トピックが画像のみを含んでいる場合、画像についての説明や描写を行う必要はありません。\n
        9.日本語で出力してください。\n
        10.出力にあなたの言葉は含めず、画像の内容のみを出力してください。\n

        """

    for image_file in Path(f"{pdf_folder}").glob("*.jpg"):

        print(f"{image_file}について処理開始")

        # 画像をバイナリデータに変換
        with Image.open(image_file) as img:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG")
            image_data = img_byte_arr.getvalue()

        # max_retries = 3  # リトライ回数
        # for attempt in range(max_retries):
        try:
            response = model.generate_content([
                {
                    "parts": [
                        {"text": prompt},  # プロンプト
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_data}} # 画像データ
                    ]
                }
            ],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", 
                response_schema=Discription
            )
            )

            # JSON をパース
            json_data = json.loads(response.text)

        except (json.JSONDecodeError, AttributeError) as e:
            print(f"JSONパースエラー: {e} ({pdf_folder}/{image_file}のパースをスキップします)")
            continue # パースエラー時はスキップ
            # return None  # パースエラー時はスキップ
            
        except google.api_core.exceptions.DeadlineExceeded as e:
            print(f"タイムアウト発生 : {e}({pdf_folder}/{image_file}のパースをスキップします)")
            continue # パースエラー時はスキップ

        # 保存先ディレクトリ
        # output_dir = f"data/vector_db/{pdf_folder_end}"
        output_dir = f"validation/vector_db/{pdf_folder_end}"
        os.makedirs(output_dir, exist_ok=True)  # ディレクトリがなければ作成

        # ファイル名を画像のベース名から生成
        json_filename = os.path.join(output_dir, f"{image_file.stem}.json")

        # JSON をファイルに保存
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        print(f"処理結果を {json_filename} に保存しました。")

if __name__ == "__main__":
    image_folder = Path("data/images")
    
    for image_file in image_folder.glob("*.jpg"):
        segments = segment_image_with_gemini(str(image_file))
        output_file = Path("data/vector_db") / f"{image_file.stem}.json"
        with open(output_file, "w") as f:
            json.dump(segments, f, indent=4)
