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
    "E5-small": "intfloat/multilingual-e5-small",
    "E5-base": "intfloat/multilingual-e5-base",
}

chunk_size = 150
overlap = 10


# ============================================================
# Complex Indonesian questions
# ============================================================

indonesian_questions = [
    {
        "id": "Q6_ID",
        "question": (
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
        "id": "Q7_ID",
        "question": (
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
        "id": "Q8_ID",
        "question": (
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
# Complex English questions
# ============================================================

english_questions = [
    {
        "id": "Q6_EN",
        "question": (
            "Explain the difference between temperature and heat, "
            "how they are related, and how heat flows between objects "
            "with different temperatures."
        ),
        "targets": [
            "suhu adalah ukuran derajat atau tingkat panas suatu benda",
            "Kalor tidak sama dengan suhu",
            "Kalor secara alamiah mengalir dari benda bersuhu lebih tinggi",
        ],
    },
    {
        "id": "Q7_EN",
        "question": (
            "How should temperature be measured accurately, "
            "why is the sense of touch not sufficient to determine "
            "temperature accurately, and what instrument should be used?"
        ),
        "targets": [
            "indra peraba bukan pengukur tingkat panas yang baik",
            "Alat yang digunakan untuk mengukur suhu dinamakan termometer",
        ],
    },
    {
        "id": "Q8_EN",
        "question": (
            "Why can two different objects experience different "
            "temperature changes even when they receive heat, "
            "and how is this related to the specific heat capacity "
            "of a material?"
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


def evaluate_questions(
    model,
    index,
    chunks,
    test_questions,
    language_name,
):

    print()
    print("=" * 100)
    print(f"{language_name.upper()} COMPLEX QUESTIONS")
    print("=" * 100)

    question_results = []

    for test in test_questions:

        question_id = test["id"]
        question = test["question"]
        targets = test["targets"]

        print()
        print(f"{question_id}: {question}")

        # ----------------------------------------------------
        # Create query embedding
        # ----------------------------------------------------

        question_embedding = model.encode(
            [f"query: {question}"],
            normalize_embeddings=True
        )

        # ----------------------------------------------------
        # Search all chunks
        # ----------------------------------------------------

        scores, indices = index.search(
            question_embedding,
            len(chunks)
        )

        target_ranks = []

        # ----------------------------------------------------
        # Check each evidence target
        # ----------------------------------------------------

        for target_number, target_text in enumerate(
            targets,
            start=1
        ):

            target_indices = find_target_indices(
                chunks,
                target_text
            )

            if not target_indices:
                print(
                    f"  Target {target_number}: "
                    f"NOT FOUND IN CHUNKS"
                )

                target_ranks.append(None)
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

        total_targets = len(targets)

        print(
            f"  Top-3 coverage: "
            f"{top3_hits}/{total_targets}"
        )

        print(
            f"  Top-5 coverage: "
            f"{top5_hits}/{total_targets}"
        )

        print(
            f"  Mean target rank: "
            f"{mean_target_rank:.2f}"
        )

        question_results.append(
            {
                "id": question_id,
                "ranks": target_ranks,
                "top3_hits": top3_hits,
                "top5_hits": top5_hits,
                "total_targets": total_targets,
            }
        )

    # --------------------------------------------------------
    # Calculate overall language result
    # --------------------------------------------------------

    total_top3 = sum(
        result["top3_hits"]
        for result in question_results
    )

    total_top5 = sum(
        result["top5_hits"]
        for result in question_results
    )

    total_targets = sum(
        result["total_targets"]
        for result in question_results
    )

    all_target_ranks = []

    for result in question_results:
        for rank in result["ranks"]:
            if rank is not None:
                all_target_ranks.append(rank)

    mean_target_rank = (
        sum(all_target_ranks) / len(all_target_ranks)
        if all_target_ranks
        else None
    )

    return {
        "top3": total_top3,
        "top5": total_top5,
        "total_targets": total_targets,
        "mean_rank": mean_target_rank,
        "question_results": question_results,
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
# 2. Test each embedding model
# ============================================================

comparison_results = []
model_information = []


for model_name, model_path in embedding_models.items():

    print()
    print("=" * 110)
    print(f"EMBEDDING MODEL: {model_name}")
    print(f"{model_path}")
    print("=" * 110)

    print("Loading embedding model...")

    model = SentenceTransformer(model_path)

    print("Embedding model loaded.")


    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    embedding_dimension = model.get_embedding_dimension()

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Embedding dimension: "
        f"{embedding_dimension}"
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Model device: "
        f"{model.device}"
    )

    model_information.append(
        {
            "model": model_name,
            "parameters": total_parameters,
            "embedding_dimension": embedding_dimension,
            "device": str(model.device),
        }
    )


    # --------------------------------------------------------
    # Create curriculum embeddings
    # --------------------------------------------------------

    passages = [
        f"passage: {chunk}"
        for chunk in chunks
    ]

    chunk_embeddings = model.encode(
        passages,
        normalize_embeddings=True
    )


    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------

    dimension = chunk_embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(chunk_embeddings)


    # --------------------------------------------------------
    # Indonesian evaluation
    # --------------------------------------------------------

    indonesian_result = evaluate_questions(
        model=model,
        index=index,
        chunks=chunks,
        test_questions=indonesian_questions,
        language_name="Indonesian",
    )


    # --------------------------------------------------------
    # English evaluation
    # --------------------------------------------------------

    english_result = evaluate_questions(
        model=model,
        index=index,
        chunks=chunks,
        test_questions=english_questions,
        language_name="English",
    )


    comparison_results.append(
        {
            "model": model_name,
            "indonesian": indonesian_result,
            "english": english_result,
        }
    )


# ============================================================
# 3. Print model information
# ============================================================

print()
print("=" * 95)
print("MODEL INFORMATION")
print("=" * 95)

print(
    f"{'Model':<12} "
    f"{'Parameters':>15} "
    f"{'Embed Dim':>12} "
    f"{'Device':>10}"
)

print("-" * 95)

for info in model_information:

    print(
        f"{info['model']:<12} "
        f"{info['parameters']:>15,} "
        f"{info['embedding_dimension']:>12} "
        f"{info['device']:>10}"
    )


# ============================================================
# 4. Final comparison
# ============================================================

print()
print("=" * 115)
print("COMPLEX RETRIEVAL EMBEDDING MODEL COMPARISON")
print("=" * 115)

print(
    f"{'Model':<12} "
    f"{'Language':<12} "
    f"{'Chunk':>7} "
    f"{'Overlap':>8} "
    f"{'Top3':>10} "
    f"{'Top5':>10} "
    f"{'Mean Rank':>12}"
)

print("-" * 115)


for result in comparison_results:

    for language_key, language_label in [
        ("indonesian", "Indonesian"),
        ("english", "English"),
    ]:

        language_result = result[language_key]

        print(
            f"{result['model']:<12} "
            f"{language_label:<12} "
            f"{chunk_size:>7} "
            f"{overlap:>8} "
            f"{str(language_result['top3']) + '/' + str(language_result['total_targets']):>10} "
            f"{str(language_result['top5']) + '/' + str(language_result['total_targets']):>10} "
            f"{language_result['mean_rank']:>12.2f}"
        )