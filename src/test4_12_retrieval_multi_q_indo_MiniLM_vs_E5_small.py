from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

pdf_path = Path(
    r"D:\Offline_AI_Teacher_Assistant\curriculum\student\Grade 7 - Science - Student Handbook - Chapter 3.pdf"
)

embedding_models = {
    "MiniLM": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "E5": "intfloat/multilingual-e5-small",
}

chunk_sizes = [80, 100, 120, 150, 180]
overlaps = [10, 20, 30]


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


# ============================================================
# Helper functions
# ============================================================

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


def find_target_indices(chunks, target_text):
    target_indices = []

    for i, chunk in enumerate(chunks):
        if target_text.lower() in chunk.lower():
            target_indices.append(i)

    return target_indices


# ============================================================
# 1. Extract curriculum text
# ============================================================

full_text = extract_text_from_pdf(pdf_path)


# ============================================================
# 2. Test each embedding model
# ============================================================

all_results = []


for model_name, model_path in embedding_models.items():

    print()
    print("=" * 110)
    print(f"EMBEDDING MODEL: {model_name}")
    print(f"{model_path}")
    print("=" * 110)

    print("Loading embedding model...")

    model = SentenceTransformer(model_path)

    print("Embedding model loaded.")


    # ========================================================
    # 3. Run full chunking grid
    # ========================================================

    for chunk_size in chunk_sizes:

        for overlap in overlaps:

            chunks = split_into_chunks(
                full_text,
                chunk_size=chunk_size,
                overlap=overlap
            )


            # ------------------------------------------------
            # Prepare passages
            # ------------------------------------------------

            if model_name == "E5":
                passages = [
                    f"passage: {chunk}"
                    for chunk in chunks
                ]
            else:
                passages = chunks


            chunk_embeddings = model.encode(
                passages,
                normalize_embeddings=True
            )


            # ------------------------------------------------
            # Build FAISS index
            # ------------------------------------------------

            dimension = chunk_embeddings.shape[1]

            index = faiss.IndexFlatIP(dimension)

            index.add(chunk_embeddings)


            question_ranks = []


            print()
            print(
                f"{model_name} | "
                f"chunk={chunk_size}, "
                f"overlap={overlap}, "
                f"chunks={len(chunks)}"
            )


            # ------------------------------------------------
            # Test each question
            # ------------------------------------------------

            for test in test_questions:

                question = test["question"]
                target_text = test["target_text"]


                if model_name == "E5":
                    query_text = f"query: {question}"
                else:
                    query_text = question


                question_embedding = model.encode(
                    [query_text],
                    normalize_embeddings=True
                )


                scores, indices = index.search(
                    question_embedding,
                    len(chunks)
                )


                target_indices = find_target_indices(
                    chunks,
                    target_text
                )


                best_rank = None
                best_chunk = None
                best_score = None


                for rank, (score, chunk_index) in enumerate(
                    zip(scores[0], indices[0]),
                    start=1
                ):

                    if chunk_index in target_indices:

                        best_rank = rank
                        best_chunk = chunk_index + 1
                        best_score = score

                        break


                question_ranks.append(best_rank)


                print(
                    f"{test['id']}: "
                    f"rank={best_rank}, "
                    f"chunk={best_chunk}, "
                    f"score={best_score:.4f}"
                )


            # ------------------------------------------------
            # Calculate metrics
            # ------------------------------------------------

            valid_ranks = [
                rank
                for rank in question_ranks
                if rank is not None
            ]


            top1 = sum(
                rank <= 1
                for rank in valid_ranks
            )

            top3 = sum(
                rank <= 3
                for rank in valid_ranks
            )

            top5 = sum(
                rank <= 5
                for rank in valid_ranks
            )

            mean_rank = (
                sum(valid_ranks) / len(valid_ranks)
                if valid_ranks
                else None
            )


            all_results.append(
                {
                    "model": model_name,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "ranks": question_ranks,
                    "top1": top1,
                    "top3": top3,
                    "top5": top5,
                    "mean_rank": mean_rank,
                }
            )


# ============================================================
# 4. Print full comparison table
# ============================================================

print()
print("=" * 135)
print("EMBEDDING MODEL COMPARISON")
print("=" * 135)


print(
    f"{'Model':<10} "
    f"{'Chunk':>6} "
    f"{'Overlap':>8} "
    f"{'Q1':>5} "
    f"{'Q2':>5} "
    f"{'Q3':>5} "
    f"{'Q4':>5} "
    f"{'Q5':>5} "
    f"{'Top1':>6} "
    f"{'Top3':>6} "
    f"{'Top5':>6} "
    f"{'Mean':>8}"
)

print("-" * 135)


for result in all_results:

    ranks = result["ranks"]

    print(
        f"{result['model']:<10} "
        f"{result['chunk_size']:>6} "
        f"{result['overlap']:>8} "
        f"{ranks[0]:>5} "
        f"{ranks[1]:>5} "
        f"{ranks[2]:>5} "
        f"{ranks[3]:>5} "
        f"{ranks[4]:>5} "
        f"{result['top1']:>6} "
        f"{result['top3']:>6} "
        f"{result['top5']:>6} "
        f"{result['mean_rank']:>8.2f}"
    )


# ============================================================
# 5. Print best configuration per model
# ============================================================

print()
print("=" * 90)
print("BEST CONFIGURATION PER MODEL")
print("=" * 90)


for model_name in embedding_models:

    model_results = [
        result
        for result in all_results
        if result["model"] == model_name
    ]


    best_result = sorted(
        model_results,
        key=lambda result: (
            -result["top3"],
            -result["top1"],
            result["mean_rank"],
        )
    )[0]


    print()
    print(f"{model_name}")

    print(
        f"  Chunk size: {best_result['chunk_size']}"
    )

    print(
        f"  Overlap: {best_result['overlap']}"
    )

    print(
        f"  Top1: {best_result['top1']}/5"
    )

    print(
        f"  Top3: {best_result['top3']}/5"
    )

    print(
        f"  Top5: {best_result['top5']}/5"
    )

    print(
        f"  Mean rank: {best_result['mean_rank']:.2f}"
    )