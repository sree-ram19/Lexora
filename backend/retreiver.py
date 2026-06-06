from vectorstore import search

def retrieve_chunks(query):
    results = search(query)

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i]["page"]
        })

    return chunks