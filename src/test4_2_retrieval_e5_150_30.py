from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


pdf_path = Path(
    r"D:\Offline_AI_Teacher_Assistant\curriculum\student\Grade 7 - Science - Student Handbook - Chapter 3.pdf"
)

chunk_size = 150
overlap = 30
top_k = 3

embedding_model_name = "intfloat/multilingual-e5-small"


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


# 1. Extract and chunk the curriculum
full_text = extract_text_from_pdf(pdf_path)

chunks = split_into_chunks(
    full_text,
    chunk_size=chunk_size,
    overlap=overlap
)

print(f"Total chunks: {len(chunks)}")


# 2. Load the embedding model
print("Loading embedding model...")

model = SentenceTransformer(embedding_model_name)

print("Embedding model loaded.")


# 3. Create embeddings for all curriculum chunks
#
# normalize_embeddings=True makes every vector have length 1.
# This lets FAISS inner-product search behave as cosine similarity.
passages = [f"passage: {chunk}" for chunk in chunks]

chunk_embeddings = model.encode(
    passages,
    normalize_embeddings=True
)

print(f"Embedding shape: {chunk_embeddings.shape}")


# 4. Create a FAISS index
dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(chunk_embeddings)

print(f"Vectors stored in FAISS: {index.ntotal}")


# 5. Ask one test question
question = "Apa yang dimaksud dengan suhu?"

print()
print(f"QUESTION: {question}")


# 6. Convert the question into an embedding
question_embedding = model.encode(
    [f"query: {question}"],
    normalize_embeddings=True
)


# 7. Search for the closest chunks
# 8earch ALL chunks (for now) so we can inspect the complete ranking
scores, indices = index.search(
    question_embedding,
    len(chunks)
)


# 8. Print the results
print()
print(f"TOP {top_k} RESULTS")
print("=" * 80)

for rank, (score, chunk_index) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):
    print(f"RANK {rank}")
    print(f"Chunk: {chunk_index + 1}")
    print(f"Similarity score: {score:.4f}")
    print("-" * 80)
    print(chunks[chunk_index])
    print()
    print("=" * 80)


# Check specifically where Chunk 5 ranks
target_chunk = 5
target_index = target_chunk - 1

for rank, (score, chunk_index) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):
    if chunk_index == target_index:
        print()
        print("=" * 80)
        print(f"CHUNK {target_chunk} SPECIFIC CHECK")
        print(f"Overall rank: {rank}")
        print(f"Similarity score: {score:.4f}")
        print("-" * 80)
        print(chunks[target_index])
        print("=" * 80)
        break