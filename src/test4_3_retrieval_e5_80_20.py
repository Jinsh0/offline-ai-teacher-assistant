from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


pdf_path = Path(
    r"D:\Offline_AI_Teacher_Assistant\curriculum\student\Grade 7 - Science - Student Handbook - Chapter 3.pdf"
)

chunk_size = 80
overlap = 20
top_k = 3

embedding_model_name = "intfloat/multilingual-e5-small"


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages)


def split_into_chunks(text, chunk_size=80, overlap=20):
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
print(f"Chunk size: {chunk_size}")
print(f"Overlap: {overlap}")


# 2. Load the embedding model
print()
print("Loading embedding model...")

model = SentenceTransformer(embedding_model_name)

print("Embedding model loaded.")


# 3. Create embeddings for curriculum chunks
passages = [f"passage: {chunk}" for chunk in chunks]

chunk_embeddings = model.encode(
    passages,
    normalize_embeddings=True
)

print(f"Embedding shape: {chunk_embeddings.shape}")


# 4. Create FAISS index
dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(chunk_embeddings)

print(f"Vectors stored in FAISS: {index.ntotal}")


# 5. Test question
question = "Apa yang dimaksud dengan suhu?"

print()
print(f"QUESTION: {question}")


# 6. Create question embedding
question_embedding = model.encode(
    [f"query: {question}"],
    normalize_embeddings=True
)


# 7. Search ALL chunks
# We search all so we can determine the exact rank
# of the expected definition passage.
scores, indices = index.search(
    question_embedding,
    len(chunks)
)


# 8. Print only the Top K results
print()
print(f"TOP {top_k} RESULTS")
print("=" * 80)

for rank, (score, chunk_index) in enumerate(
    zip(scores[0][:top_k], indices[0][:top_k]),
    start=1
):
    print(f"RANK {rank}")
    print(f"Chunk: {chunk_index + 1}")
    print(f"Similarity score: {score:.4f}")
    print("-" * 80)
    print(chunks[chunk_index])
    print()
    print("=" * 80)


# 9. Automatically find the chunk containing the expected definition
target_text = "Apa yang Dimaksud dengan Suhu?"

target_index = None

for i, chunk in enumerate(chunks):
    if target_text.lower() in chunk.lower():
        target_index = i
        break


if target_index is not None:

    for rank, (score, chunk_index) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):
        if chunk_index == target_index:
            print()
            print("=" * 80)
            print("EXPECTED DEFINITION CHUNK")
            print(f"Chunk: {target_index + 1}")
            print(f"Overall rank: {rank}")
            print(f"Similarity score: {score:.4f}")
            print("-" * 80)
            print(chunks[target_index])
            print("=" * 80)
            break

else:
    print()
    print("Expected definition text was not found in any chunk.")