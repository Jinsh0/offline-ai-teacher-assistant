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

embedding_model_name = "intfloat/multilingual-e5-small"

chunk_size = 150
overlap = 10


# ============================================================
# Test questions
#
# Each English question has a manually translated Indonesian
# equivalent.
#
# The curriculum targets remain Indonesian.
# ============================================================

test_questions = [
    {
        "id": "Q6",
        "english_question": (
            "Explain the difference between temperature and heat, "
            "how they are related, and how heat flows between objects "
            "with different temperatures."
        ),
        "indonesian_question": (
            "Jelaskan perbedaan suhu dan kalor, bagaimana keduanya "
            "saling berhubungan, dan bagaimana kalor berpindah "
            "antara benda yang memiliki suhu berbeda."
        ),
        "targets": [
            "suhu adalah ukuran derajat atau tingkat panas suatu benda",
            "Kalor tidak sama dengan suhu",
            "Kalor secara alamiah mengalir dari benda bersuhu lebih tinggi",
        ],
    },
    {
        "id": "Q7",
        "english_question": (
            "How should temperature be measured accurately, "
            "why is the sense of touch not sufficient to determine "
            "temperature accurately, and what instrument should be used?"
        ),
        "indonesian_question": (
            "Bagaimana cara mengukur suhu dengan benar, mengapa indra "
            "peraba tidak cukup untuk menentukan suhu secara akurat, "
            "dan alat apa yang seharusnya digunakan?"
        ),
        "targets": [
            "indra peraba bukan pengukur tingkat panas yang baik",
            "Alat yang digunakan untuk mengukur suhu dinamakan termometer",
        ],
    },
    {
        "id": "Q8",
        "english_question": (
            "Why can two different objects experience different "
            "temperature changes even when they receive heat, "
            "and how is this related to the specific heat capacity "
            "of a material?"
        ),
        "indonesian_question": (
            "Mengapa dua benda yang berbeda dapat mengalami perubahan "
            "suhu yang berbeda walaupun menerima kalor, dan apa hubungan "
            "hal tersebut dengan kalor jenis suatu bahan?"
        ),
        "targets": [
            "Air termasuk zat yang memiliki kalor jenis tinggi",
            "Kalor jenis adalah jumlah energi panas yang diperlukan oleh 1 kg bahan",
        ],
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


def find_best_target_rank(scores, indices, target_indices):
    for rank, (score, chunk_index) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):
        if chunk_index in target_indices:
            return rank, chunk_index + 1, score

    return None, None, None


def evaluate_question(
    model,
    index,
    chunks,
    question,
    targets,
):
    question_embedding = model.encode(
        [f"query: {question}"],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        question_embedding,
        len(chunks)
    )

    target_ranks = []

    for target_number, target_text in enumerate(
        targets,
        start=1
    ):
        target_indices = find_target_indices(
            chunks,
            target_text
        )

        if not target_indices:
            target_ranks.append(None)

            print(
                f"  Target {target_number}: "
                f"NOT FOUND IN CHUNKS"
            )

            continue

        best_rank, best_chunk, best_score = (
            find_best_target_rank(
                scores,
                indices,
                target_indices
            )
        )

        target_ranks.append(best_rank)

        print(
            f"  Target {target_number}: "
            f"rank={best_rank}, "
            f"chunk={best_chunk}, "
            f"score={best_score:.4f}"
        )

    valid_ranks = [
        rank
        for rank in target_ranks
        if rank is not None
    ]

    top3_hits = sum(
        rank <= 3
        for rank in valid_ranks
    )

    top5_hits = sum(
        rank <= 5
        for rank in valid_ranks
    )

    mean_target_rank = (
        sum(valid_ranks) / len(valid_ranks)
        if valid_ranks
        else None
    )

    return {
        "ranks": target_ranks,
        "top3_hits": top3_hits,
        "top5_hits": top5_hits,
        "mean_target_rank": mean_target_rank,
        "total_targets": len(targets),
    }


# ============================================================
# 1. Extract and chunk curriculum
# ============================================================

full_text = extract_text_from_pdf(pdf_path)

chunks = split_into_chunks(
    full_text,
    chunk_size=chunk_size,
    overlap=overlap
)

print(
    f"Chunk size={chunk_size}, "
    f"overlap={overlap}, "
    f"total chunks={len(chunks)}"
)


# ============================================================
# 2. Load embedding model
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(
    embedding_model_name
)

print("Embedding model loaded.")


# ============================================================
# 3. Create curriculum embeddings
# ============================================================

passages = [
    f"passage: {chunk}"
    for chunk in chunks
]

chunk_embeddings = model.encode(
    passages,
    normalize_embeddings=True
)


# ============================================================
# 4. Create FAISS index
# ============================================================

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(chunk_embeddings)


# ============================================================
# 5. Compare English vs translated Indonesian queries
# ============================================================

english_results = []
indonesian_results = []


for test in test_questions:

    question_id = test["id"]
    english_question = test["english_question"]
    indonesian_question = test["indonesian_question"]
    targets = test["targets"]


    # --------------------------------------------------------
    # A. Original English query
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(f"{question_id} - ORIGINAL ENGLISH QUERY")
    print("=" * 90)

    print(english_question)
    print()

    english_result = evaluate_question(
        model=model,
        index=index,
        chunks=chunks,
        question=english_question,
        targets=targets,
    )

    english_results.append(
        english_result
    )

    print(
        f"  Top-3 coverage: "
        f"{english_result['top3_hits']}/"
        f"{english_result['total_targets']}"
    )

    print(
        f"  Top-5 coverage: "
        f"{english_result['top5_hits']}/"
        f"{english_result['total_targets']}"
    )

    print(
        f"  Mean target rank: "
        f"{english_result['mean_target_rank']:.2f}"
    )


    # --------------------------------------------------------
    # B. Translated Indonesian query
    # --------------------------------------------------------

    print()
    print("-" * 90)
    print(f"{question_id} - TRANSLATED INDONESIAN QUERY")
    print("-" * 90)

    print(indonesian_question)
    print()

    indonesian_result = evaluate_question(
        model=model,
        index=index,
        chunks=chunks,
        question=indonesian_question,
        targets=targets,
    )

    indonesian_results.append(
        indonesian_result
    )

    print(
        f"  Top-3 coverage: "
        f"{indonesian_result['top3_hits']}/"
        f"{indonesian_result['total_targets']}"
    )

    print(
        f"  Top-5 coverage: "
        f"{indonesian_result['top5_hits']}/"
        f"{indonesian_result['total_targets']}"
    )

    print(
        f"  Mean target rank: "
        f"{indonesian_result['mean_target_rank']:.2f}"
    )


# ============================================================
# 6. Overall comparison
# ============================================================

print()
print("=" * 105)
print("QUERY TRANSLATION COMPARISON")
print("=" * 105)

print(
    f"{'Mode':<25} "
    f"{'Top3':>10} "
    f"{'Top5':>10} "
    f"{'Mean Rank':>12}"
)

print("-" * 105)


def calculate_overall(results):
    top3_total = sum(
        result["top3_hits"]
        for result in results
    )

    top5_total = sum(
        result["top5_hits"]
        for result in results
    )

    total_targets = sum(
        result["total_targets"]
        for result in results
    )

    all_ranks = []

    for result in results:
        for rank in result["ranks"]:
            if rank is not None:
                all_ranks.append(rank)

    mean_rank = (
        sum(all_ranks) / len(all_ranks)
        if all_ranks
        else None
    )

    return (
        top3_total,
        top5_total,
        total_targets,
        mean_rank,
    )


english_top3, english_top5, total_targets, english_mean = (
    calculate_overall(
        english_results
    )
)

indo_top3, indo_top5, _, indo_mean = (
    calculate_overall(
        indonesian_results
    )
)


print(
    f"{'Original English':<25} "
    f"{str(english_top3) + '/' + str(total_targets):>10} "
    f"{str(english_top5) + '/' + str(total_targets):>10} "
    f"{english_mean:>12.2f}"
)

print(
    f"{'Translated Indonesian':<25} "
    f"{str(indo_top3) + '/' + str(total_targets):>10} "
    f"{str(indo_top5) + '/' + str(total_targets):>10} "
    f"{indo_mean:>12.2f}"
)