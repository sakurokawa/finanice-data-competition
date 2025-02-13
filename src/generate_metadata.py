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
from utils.schema import Metadata_reportname, Metadata_contents
from pathlib import Path

@retry.Retry()
def generate_metadata_with_gemini(pdf_folder: str, pdf_except_prefix: str, vector_prefix: str):
    
    #pathの末尾を取得(ex.1_pdf)
    # pdf_folder_end = pdf_folder.removeprefix("data/images/")
    # pdf_folder_end = pdf_folder.removeprefix("validation/images/")
    pdf_folder_end = str(pdf_folder).removeprefix(pdf_except_prefix)


    # 環境変数読み込み
    load_dotenv(verbose=True)
    genai.configure(api_key = os.getenv('GEMINI_API_KEY '))
    model = genai.GenerativeModel("gemini-1.5-flash")

    print(f"{pdf_folder}について処理を開始します")

    prompt_reportname = """
        
        次の画像は、ある会社の統合報告書あるいはそれに類するレポートの一部です。\n
        画像にある内容を次の指示に従って漏れがないようにJSON形式で出力してください。\n
        1.統合報告書あるいはレポートの名称を会社名を含めた形で、JSON内のtitleに設定してください。\n
        2.画像内で言及されている会社名については、JSON内のcompanyに設定してください。\n
        3.日本語で出力してください。\n
        4.出力にあなたの言葉は含めず、画像から得られた情報のみを使って出力してください。\n

        """
    
    prompt_contents = """
        
        次の画像は、ある会社の統合報告書あるいはそれに類するレポートの一部です。\n
        画像にある内容に関するメタデータを次の指示に従って漏れがないようにJSON形式で出力してください。\n
        1.統合報告書あるいはレポートに記載されている内容について、トピックに分けてまとめたものをJSON内のcontentに出力してください。\n
        2.contentは全体で100文字程度で簡潔に出力してください。トピックが複数存在する場合には、改行は使用せず、句点(、)で区切って出力してください。\n
        3.pageには画像上で記載されているページ数を記載してください。ページが複数に分かれる場合には、句点(、)で区切って出力してください。\n
        4.日本語で出力してください。\n
        5.出力にあなたの言葉は含めず、画像から得られた情報のみを使って出力してください。\n

        """
    
    #レポート名を取得
    for image_file in Path(f"{pdf_folder}").glob("page_1.jpg"):

        print(f"{image_file}について処理開始")

        # 画像をバイナリデータに変換
        with Image.open(image_file) as img:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG")
            image_data = img_byte_arr.getvalue()

        try:
            response = model.generate_content([
                {
                    "parts": [
                        {"text": prompt_reportname},  # プロンプト
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_data}} # 画像データ
                    ]
                }
            ],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", 
                response_schema=Metadata_reportname
            )
            )
            # JSON をパース
            reportname_data = json.loads(response.text)
            print(f"レポート名は{reportname_data}")

        except (json.JSONDecodeError, AttributeError) as e:
            print(f"JSONパースエラー: {e} ({pdf_folder}/{image_file}のパースをスキップします)")
            continue # パースエラー時はスキップ
            # return None  # パースエラー時はスキップ
            
        except google.api_core.exceptions.DeadlineExceeded as e:
            print(f"タイムアウト発生 : {e}({pdf_folder}/{image_file}のパースをスキップします)")
            continue # パースエラー時はスキップ

    ############################

    # コンテンツ名を取得
    for image_file in Path(f"{pdf_folder}").glob("*.jpg"):

        # 返却データの初期化
        json_data = {}

        print(f"{image_file}について処理開始")

        # 画像をバイナリデータに変換
        with Image.open(image_file) as img:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG")
            image_data = img_byte_arr.getvalue()

        try:
            response = model.generate_content([
                {
                    "parts": [
                        {"text": prompt_contents},  # プロンプト
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_data}} # 画像データ
                    ]
                }
            ],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", 
                response_schema=Metadata_contents
            )
            )

            # JSON をパース
            contents_data = json.loads(response.text)

        except (json.JSONDecodeError, AttributeError) as e:
            print(f"JSONパースエラー: {e} ({pdf_folder}/{image_file}のパースをスキップします)")
            continue # パースエラー時はスキップ
            # return None  # パースエラー時はスキップ
            
        except google.api_core.exceptions.DeadlineExceeded as e:
            print(f"タイムアウト発生 : {e}({pdf_folder}/{image_file}のパースをスキップします)")
            continue # パースエラー時はスキップ

        json_data["title"] = reportname_data["title"]
        json_data["company"] = reportname_data["company"]
        json_data["contents"] = contents_data["contents"]
        json_data["filename"] = pdf_folder_end
        json_data["page"] = contents_data["page"]
        print(json_data)
        # 保存先ディレクトリ
        # output_dir = f"data/vector_db/{pdf_folder_end}"
        # output_dir = f"validation/vector_db/{pdf_folder_end}"
        output_dir = f"{vector_prefix}{pdf_folder_end}"
        os.makedirs(output_dir, exist_ok=True)  # ディレクトリがなければ作成

        # ファイル名を画像のベース名から生成
        json_filename = os.path.join(output_dir, f"{image_file.stem}.json")

        # JSON をファイルに保存
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        print(f"処理結果を {json_filename} に保存しました。")
        print(json_data)

if __name__ == "__main__":
    image_folder = Path("data/images")
    
    for image_file in image_folder.glob("*.jpg"):
        segments = generate_metadata_with_gemini(str(image_file))
        output_file = Path("data/vector_db") / f"{image_file.stem}.json"
        with open(output_file, "w") as f:
            json.dump(segments, f, indent=4)
