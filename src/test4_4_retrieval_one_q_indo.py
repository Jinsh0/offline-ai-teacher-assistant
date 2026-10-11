from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


pdf_path = Path(
    r"D:\Offline_AI_Teacher_Assistant\curriculum\student\Grade 7 - Science - Student Handbook - Chapter 3.pdf"
)

embedding_model_name = "intfloat/multilingual-e5-small"

question = "Apa yang dimaksud dengan suhu?"

target_text = "Apa yang Dimaksud dengan Suhu?"


# Chunk sizes and overlaps we want to test
chunk_sizes = [80, 100, 120, 150, 180]
overlaps = [10, 20, 30]


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


# 1. Extract PDF once
full_text = extract_text_from_pdf(pdf_path)


# 2. Load embedding model once
print("Loading embedding model...")

model = SentenceTransformer(embedding_model_name)

print("Embedding model loaded.")


# 3. Create question embedding once
question_embedding = model.encode(
    [f"query: {question}"],
    normalize_embeddings=True
)


results = []


# 4. Test every chunk-size / overlap combination
for chunk_size in chunk_sizes:

    for overlap in overlaps:

        # Skip invalid combinations
        if overlap >= chunk_size:
            continue

        chunks = split_into_chunks(
            full_text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        # Find the chunk containing the expected definition
        target_index = None

        for i, chunk in enumerate(chunks):
            if target_text.lower() in chunk.lower():
                target_index = i
                break

        if target_index is None:
            print(
                f"Target text not found for "
                f"chunk_size={chunk_size}, overlap={overlap}"
            )
            continue


        # Create passage embeddings
        passages = [
            f"passage: {chunk}"
            for chunk in chunks
        ]

        chunk_embeddings = model.encode(
            passages,
            normalize_embeddings=True
        )


        # Create FAISS index
        dimension = chunk_embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(chunk_embeddings)


        # Search all chunks
        scores, indices = index.search(
            question_embedding,
            len(chunks)
        )


        # Find rank of expected definition chunk
        target_rank = None
        target_score = None

        for rank, (score, chunk_index) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):
            if chunk_index == target_index:
                target_rank = rank
                target_score = score
                break


        results.append(
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "total_chunks": len(chunks),
                "target_chunk": target_index + 1,
                "target_rank": target_rank,
                "target_score": target_score,
            }
        )


# 5. Sort best rank first
results.sort(
    key=lambda x: (
        x["target_rank"],
        -x["target_score"]
    )
)


# 6. Print summary table
print()
print("=" * 85)
print("GRID SEARCH RESULTS")
print("=" * 85)

print(
    f"{'Chunk':>7} "
    f"{'Overlap':>8} "
    f"{'Chunks':>8} "
    f"{'Target':>8} "
    f"{'Rank':>6} "
    f"{'Score':>8}"
)

print("-" * 85)

for result in results:

    print(
        f"{result['chunk_size']:>7} "
        f"{result['overlap']:>8} "
        f"{result['total_chunks']:>8} "
        f"{result['target_chunk']:>8} "
        f"{result['target_rank']:>6} "
        f"{result['target_score']:>8.4f}"
    )