from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


pdf_path = Path(
    r"D:\Offline_AI_Teacher_Assistant\curriculum\student\Grade 7 - Science - Student Handbook - Chapter 3.pdf"
)

embedding_model_name = "intfloat/multilingual-e5-small"

test_questions = [
    {
        "id": "Q1",
        "question": "Apa yang dimaksud dengan suhu?",
        "target_text": "suhu adalah ukuran derajat atau tingkat panas suatu benda",
    },
    {
        "id": "Q2",
        "question": "Apa alat yang digunakan untuk mengukur suhu?",
        "target_text": "Alat yang digunakan untuk mengukur suhu dinamakan termometer",
    },
    {
        "id": "Q3",
        "question": "Apa yang dimaksud dengan skala suhu?",
        "target_text": "Skala suhu menunjukkan seberapa besar nilai suhu benda",
    },
    {
        "id": "Q4",
        "question": "Bagaimana kalor berpindah secara alamiah?",
        "target_text": "Kalor secara alamiah mengalir dari benda bersuhu lebih tinggi",
    },
    {
        "id": "Q5",
        "question": "Apa yang dimaksud dengan kalor jenis?",
        "target_text": "Kalor jenis adalah jumlah energi panas yang diperlukan oleh 1 kg bahan",
    },
]


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


results = []


# 3. Test every chunk-size / overlap combination
for chunk_size in chunk_sizes:

    for overlap in overlaps:

        if overlap >= chunk_size:
            continue

        chunks = split_into_chunks(
            full_text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        # Create passage embeddings once for this chunk configuration
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

        question_ranks = []

        print()
        print(
            f"Testing chunk_size={chunk_size}, "
            f"overlap={overlap}, "
            f"total_chunks={len(chunks)}"
        )


        # 4. Test all questions
        for test in test_questions:

            question_id = test["id"]
            question = test["question"]
            target_text = test["target_text"]


            # Find ALL chunks containing the expected target text
            target_indices = []

            for i, chunk in enumerate(chunks):
                if target_text.lower() in chunk.lower():
                    target_indices.append(i)


            if not target_indices:
                print(
                    f"{question_id}: target text not found"
                )

                question_ranks.append(None)
                continue


            # Create embedding for this question
            question_embedding = model.encode(
                [f"query: {question}"],
                normalize_embeddings=True
            )


            # Search all chunks
            scores, indices = index.search(
                question_embedding,
                len(chunks)
            )


            # Find the BEST rank among all valid target chunks
            best_rank = None
            best_score = None
            best_chunk = None

            for rank, (score, chunk_index) in enumerate(
                zip(scores[0], indices[0]),
                start=1
            ):

                if chunk_index in target_indices:

                    best_rank = rank
                    best_score = score
                    best_chunk = chunk_index + 1
                    break


            question_ranks.append(best_rank)

            print(
                f"{question_id}: "
                f"rank={best_rank}, "
                f"chunk={best_chunk}, "
                f"score={best_score:.4f}"
            )


        # 5. Calculate metrics for this configuration

        valid_ranks = [
            rank
            for rank in question_ranks
            if rank is not None
        ]

        top1 = sum(rank <= 1 for rank in valid_ranks)
        top3 = sum(rank <= 3 for rank in valid_ranks)
        top5 = sum(rank <= 5 for rank in valid_ranks)

        mean_rank = (
            sum(valid_ranks) / len(valid_ranks)
            if valid_ranks
            else None
        )


        results.append(
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "total_chunks": len(chunks),
                "ranks": question_ranks,
                "top1": top1,
                "top3": top3,
                "top5": top5,
                "mean_rank": mean_rank,
            }
        )


# 6. Sort results
#
# Priority:
# 1. More Top-3 successes
# 2. More Top-1 successes
# 3. Lower mean rank
results.sort(
    key=lambda x: (
        -x["top3"],
        -x["top1"],
        x["mean_rank"]
    )
)


# 7. Print summary table

print()
print("=" * 115)
print("MULTI-QUESTION GRID SEARCH RESULTS")
print("=" * 115)

print(
    f"{'Chunk':>6} "
    f"{'Overlap':>7} "
    f"{'Chunks':>7} "
    f"{'Q1':>5} "
    f"{'Q2':>5} "
    f"{'Q3':>5} "
    f"{'Q4':>5} "
    f"{'Q5':>5} "
    f"{'Top1':>6} "
    f"{'Top3':>6} "
    f"{'Top5':>6} "
    f"{'Mean':>7}"
)

print("-" * 115)


for result in results:

    ranks = result["ranks"]

    rank_values = [
        str(rank) if rank is not None else "-"
        for rank in ranks
    ]

    print(
        f"{result['chunk_size']:>6} "
        f"{result['overlap']:>7} "
        f"{result['total_chunks']:>7} "
        f"{rank_values[0]:>5} "
        f"{rank_values[1]:>5} "
        f"{rank_values[2]:>5} "
        f"{rank_values[3]:>5} "
        f"{rank_values[4]:>5} "
        f"{result['top1']:>6} "
        f"{result['top3']:>6} "
        f"{result['top5']:>6} "
        f"{result['mean_rank']:>7.2f}"
    )