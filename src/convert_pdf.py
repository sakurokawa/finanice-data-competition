import os
from pdf2image import convert_from_path
from pathlib import Path

def convert_pdf_to_images(pdf_path: str, output_folder: str):
    pdf_path = Path(pdf_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    images = convert_from_path(pdf_path, fmt="jpeg", dpi=300)

    # 保存先ディレクトリ
    output_dir = output_folder / f"{pdf_path.stem}_pdf"
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(images):
        img_path = output_dir / f"page_{i+1}.jpg"
        img.save(img_path, "JPEG")

if __name__ == "__main__":
    input_folder = "data/pdfs"
    output_folder = "data/images"
    for pdf_file in Path(input_folder).glob("*.pdf"):
        convert_pdf_to_images(str(pdf_file), output_folder)
