from src.convert_pdf import convert_pdf_to_images
from src.segment_text import segment_image_with_gemini
from src.segment_markdown import image_to_md_with_gemini
from src.store_md_to_vectors import store_md_to_vector_db,metadata_to_vector_db
from src.store_vectors import store_segments_to_vector_db
from src.answer_queries import answer_questions ,answer_questions_second
from src.answer_md_queries import answer_questions_md, answer_questions_md_second
from src.generate_metadata import generate_metadata_with_gemini
from pathlib import Path

def main():
    # テストデータ
    # print("Step 1: Convert PDFs to Images")
    # for pdf_file in Path("data/pdfs").glob("*.pdf"):
    #     print(f"{pdf_file}の画像化開始")
    #     convert_pdf_to_images(str(pdf_file), "data/images")

    # バリデーションデータ
    # print("Step 1: Convert PDFs to Images")
    # for pdf_file in Path("validation/documents").glob("*.pdf"):
    #     print(f"{pdf_file}の画像化開始")
    #     convert_pdf_to_images(str(pdf_file), "validation/images")

    # テストデータ
    # print("Step 2: Segment Text into JSON using Gemini")
    # for pdf_folder in Path("data/images").glob("*_pdf"):
    #     segment_image_with_gemini(str(pdf_folder))

    # # バリデーションデータ
    # print("Step 2: Segment Text into JSON using Gemini")
    # for pdf_folder in Path("validation/images").glob("*_pdf"):
    #     segment_image_with_gemini(str(pdf_folder))

    # # テストデータ
    # print("Step 2: Segment Text into MarkDown using Gemini")
    # for pdf_folder in Path("data/images").glob("*_pdf"):
    #     image_to_md_with_gemini(str(pdf_folder))

    # バリデーションデータ
    # print("Step 2: Segment Text into MarkDown using Gemini")
    # for pdf_folder in Path("validation/images").glob("*_pdf"):
    #     image_to_md_with_gemini(str(pdf_folder))

    # # テストデータ
    # print("メタデータ作成")
    # for json_folder in Path("data/images").glob("*_pdf"):
    #     generate_metadata_with_gemini(json_folder,"data/images/","data/vector_db/")

    # バリデーションデータ
    # print("メタデータ作成")
    # for json_folder in Path("validation/images").glob("*_pdf"):
    #     generate_metadata_with_gemini(json_folder,"validation/images/","validation/vector_db/")

    # # テストデータ
    # print("Step 3: Store Segments to Vector DB")
    # for json_folder in Path("data/vector_db").glob("*_pdf"):
    #     store_segments_to_vector_db("data/vector_db", json_folder)

    # # バリデーションデータ
    # print("Step 3: Store Segments to Vector DB")
    # for json_folder in Path("validation/vector_db").glob("*_pdf"):
    #     store_segments_to_vector_db("validation/vector_db", json_folder)

    # テストデータ
    print("Step 3: Store Markdown & Metadata to Vector DB")
    for md_folder in Path("validation/vector_db").glob("*_pdf"):
        store_md_to_vector_db("validation/vector_db", md_folder)
        metadata_to_vector_db("validation/vector_db", md_folder)

    # バリデーションデータ
    # print("Step 3: Store Markdown & Metadata to Vector DB")
    # for md_folder in Path("validation/vector_db").glob("*_pdf"):
    #     store_md_to_vector_db("validation/vector_db", md_folder)

    # # テストデータ
    # print("Step 4: Answer Queries using Azure OpenAI")
    # answer_questions("data/query.csv", "data/vector_db", "data/prediction.csv")

    # # バリデーションデータ
    # print("Step 4: Answer Queries using Azure OpenAI")
    # answer_questions_md("validation/query.csv", "validation/vector_db", "validation/prediction.csv")

    # print("Step 5: Answer Queries using Azure OpenAI again")
    # answer_questions_second("data/query.csv", "data/vector_db", "data/prediction.csv", "data/prediction.csv")
    
    # # バリデーションデータ
    # print("Step 5: Answer Queries using Azure OpenAI again")
    # answer_questions_md_second("validation/query.csv", "validation/vector_db", "validation/prediction.csv", "validation/prediction.csv")

if __name__ == "__main__":
    main()
