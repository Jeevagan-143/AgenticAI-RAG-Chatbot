import os
import uuid

import pymupdf
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = "data/Ebook-Agentic-AI.pdf"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-rag-free")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


def extract_pdf():
    print("Reading PDF...")

    pdf = pymupdf.open(PDF_PATH)
    documents = []

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text")

        if text and text.strip():
            documents.append({
                "page": page_number,
                "text": text.strip()
            })

    pdf.close()

    print(f"Pages with text: {len(documents)}")

    return documents


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for document in documents:
        texts = splitter.split_text(document["text"])

        for text in texts:
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "page": document["page"]
            })

    print(f"Total chunks: {len(chunks)}")

    return chunks


def upload_to_pinecone(chunks):
    print("Loading FREE embedding model...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating embeddings...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    vectors = []

    for i, chunk in enumerate(chunks):
        vectors.append({
            "id": chunk["id"],
            "values": embeddings[i].tolist(),
            "metadata": {
                "text": chunk["text"],
                "page": chunk["page"],
                "source": "Ebook-Agentic-AI.pdf"
            }
        })

    print("Uploading vectors to Pinecone...")

    batch_size = 50

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

        print(
            f"Uploaded {min(i + batch_size, len(vectors))}/{len(vectors)}"
        )

    print("Pinecone upload completed!")


def main():
    documents = extract_pdf()

    if not documents:
        print("ERROR: No text extracted.")
        return

    chunks = create_chunks(documents)

    if not chunks:
        print("ERROR: No chunks created.")
        return

    upload_to_pinecone(chunks)


if __name__ == "__main__":
    main()