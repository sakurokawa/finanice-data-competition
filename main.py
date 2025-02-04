from src.convert_pdf import convert_pdf_to_images
from src.segment_text import segment_image_with_gemini
from src.store_vectors import store_segments_to_vector_db
from src.answer_queries import answer_questions ,answer_questions_second
from pathlib import Path

def main():
    print("Step 1: Convert PDFs to Images")
    for pdf_file in Path("data/pdfs").glob("*.pdf"):
        print(f"{pdf_file}の画像化開始")
        convert_pdf_to_images(str(pdf_file), "data/images")

    print("Step 2: Segment Text using Gemini")
    for image_file in Path("data/images").glob("*.jpg"):
        segment_image_with_gemini(str(image_file))

    print("Step 2: Segment Text using Gemini")
    for pdf_folder in Path("data/images").glob("*_pdf"):
        segment_image_with_gemini(str(pdf_folder))

    print("Step 3: Store Segments to Vector DB")
    for json_folder in Path("data/vector_db").glob("*_pdf"):
        store_segments_to_vector_db("data/vector_db", json_folder)

    print("Step 4: Answer Queries using Azure OpenAI")
    answer_questions("data/query.csv", "data/vector_db", "data/prediction.csv")

    print("Step 5: Answer Queries using Azure OpenAI again")
    answer_questions_second("data/query.csv", "data/vector_db", "data/prediction.csv", "data/prediction.csv")

if __name__ == "__main__":
    main()
