from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


pdf_path = Path(
    r"D:\Offline_AI_Teacher_Assistant\curriculum\student\Grade 7 - Science - Student Handbook - Chapter 3.pdf"
)

chunk_size = 150
overlap = 30

embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages)


def split_into_chunks(text, chunk_size=150, overlap=30):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# 1. Extract text
full_text = extract_text_from_pdf(pdf_path)

# 2. Create chunks
chunks = split_into_chunks(
    full_text,
    chunk_size=chunk_size,
    overlap=overlap
)

print(f"Total chunks: {len(chunks)}")


# 3. Load embedding model
print()
print("Loading embedding model...")

model = SentenceTransformer(embedding_model_name)

print("Embedding model loaded.")


# 4. Convert chunks into embeddings
print()
print("Creating embeddings...")

embeddings = model.encode(chunks)

print("Embeddings created.")


# 5. Inspect result
print()
print(f"Number of chunks: {len(chunks)}")
print(f"Embedding array shape: {embeddings.shape}")

print()
print("First chunk:")
print("-" * 80)
print(chunks[0])

print()
print("First 10 values of its embedding:")
print(embeddings[0][:10])