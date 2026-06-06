from pdfloader import extract_text
from chunker import chunk_text
from vectorstore import add_chunks
from rag import answer_question

# 1. Load PDF
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "data", "sample.pdf")

pages = extract_text(PDF_PATH)

# 2. Chunk
chunks = chunk_text(pages)

# 3. Store
add_chunks(chunks, doc_name="sample")

# 4. Ask question
while True:
    q = input("Ask: ")
    print(answer_question(q))