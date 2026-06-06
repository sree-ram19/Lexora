from groq import Groq
import os
from dotenv import load_dotenv
from retreiver import retrieve_chunks

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_question(question):
    chunks = retrieve_chunks(question)

    context = "\n\n".join(
        [f"(Page {c['page']}) {c['text']}" for c in chunks]
    )

    prompt = f"""
You are Lexora AI.

Use ONLY the context below.
If answer is not in context, say "Not found in document".

Context:
{context}

Question:
{question}

Give answer with page citations.
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content