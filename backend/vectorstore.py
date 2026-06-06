import chromadb
from embeddings import get_embedding

client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection(name="lexora")

def add_chunks(chunks, doc_name="doc1"):
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk["text"])

        collection.add(
            ids=[f"{doc_name}_{i}"],
            embeddings=[emb],
            documents=[chunk["text"]],
            metadatas=[{
                "page": chunk["page"],
                "doc": doc_name
            }]
        )


def search(query, k=5):
    q_emb = get_embedding(query)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )

    return results