from pdfloader import extract_text
from chunker import chunk_text
from vectorstore import add_chunks
from rag import answer_question

# 1. Load PDF
pages = extract_text("sample.pdf")

# 2. Chunk
chunks = chunk_text(pages)

# 3. Store
add_chunks(chunks, doc_name="sample")

# 4. Ask question
while True:
    q = input("Ask: ")
    print(answer_question(q))