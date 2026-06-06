import fitz  # PyMuPDF

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        pages_text.append({
            "page": page_num + 1,
            "text": text
        })

    return pages_text


if __name__ == "__main__":
    data = extract_text("sample.pdf")
    print(data[0])