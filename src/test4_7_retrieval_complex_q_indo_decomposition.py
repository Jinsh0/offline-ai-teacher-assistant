from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


pdf_path = Path(
    r"D:\Offline_AI_Teacher_Assistant\curriculum\student\Grade 7 - Science - Student Handbook - Chapter 3.pdf"
)

embedding_model_name = "intfloat/multilingual-e5-small"

chunk_size = 150
overlap = 10
top_k = 3


main_question = (
    "Jelaskan perbedaan suhu dan kalor, bagaimana keduanya saling "
    "berhubungan, dan bagaimana kalor berpindah antara benda yang "
    "memiliki suhu berbeda."
)


sub_questions = [
    "Apa yang dimaksud dengan suhu?",
    "Apa yang dimaksud dengan kalor?",
    "Bagaimana kalor berpindah antara benda yang memiliki suhu berbeda?",
]


targets = [
    "suhu adalah ukuran derajat atau tingkat panas suatu benda",
    "Kalor tidak sama dengan suhu",
    "Kalor secara alamiah mengalir dari benda bersuhu lebih tinggi",
]


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages)


def split_into_chunks(text, chunk_size, overlap):
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


# 1. Extract and chunk PDF
full_text = extract_text_from_pdf(pdf_path)

chunks = split_into_chunks(
    full_text,
    chunk_size=chunk_size,
    overlap=overlap
)

print(f"Total chunks: {len(chunks)}")


# 2. Load embedding model
print("Loading embedding model...")

model = SentenceTransformer(embedding_model_name)

print("Embedding model loaded.")


# 3. Create passage embeddings
passages = [
    f"passage: {chunk}"
    for chunk in chunks
]

chunk_embeddings = model.encode(
    passages,
    normalize_embeddings=True
)


# 4. Create FAISS index
dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(chunk_embeddings)


# 5. Retrieve Top K for every sub-question
retrieved_chunk_indices = set()

print()
print("=" * 90)
print("QUERY DECOMPOSITION")
print("=" * 90)

print()
print("MAIN QUESTION:")
print(main_question)


for i, sub_question in enumerate(sub_questions, start=1):

    print()
    print("-" * 90)
    print(f"SUB-QUESTION {i}: {sub_question}")

    question_embedding = model.encode(
        [f"query: {sub_question}"],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        question_embedding,
        top_k
    )

    for rank, (score, chunk_index) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):
        retrieved_chunk_indices.add(chunk_index)

        print(
            f"  Rank {rank}: "
            f"Chunk {chunk_index + 1}, "
            f"Score {score:.4f}"
        )


# 6. Print all unique retrieved chunks
print()
print("=" * 90)
print("MERGED UNIQUE RETRIEVED CHUNKS")
print("=" * 90)

sorted_retrieved = sorted(retrieved_chunk_indices)

for chunk_index in sorted_retrieved:

    print()
    print(f"CHUNK {chunk_index + 1}")
    print("-" * 90)
    print(chunks[chunk_index])


# 7. Check whether each required target was retrieved
print()
print("=" * 90)
print("TARGET COVERAGE")
print("=" * 90)

hits = 0

for target_number, target_text in enumerate(targets, start=1):

    found = False
    found_chunk = None

    for chunk_index in retrieved_chunk_indices:

        if target_text.lower() in chunks[chunk_index].lower():
            found = True
            found_chunk = chunk_index + 1
            break

    if found:
        hits += 1

        print(
            f"Target {target_number}: FOUND "
            f"in Chunk {found_chunk}"
        )

    else:
        print(
            f"Target {target_number}: NOT FOUND"
        )


print()
print(f"Total target coverage: {hits}/{len(targets)}")