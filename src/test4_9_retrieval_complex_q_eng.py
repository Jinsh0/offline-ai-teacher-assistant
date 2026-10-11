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

chunk_sizes = [80, 100, 120, 150, 180]
overlaps = [10, 20, 30]


test_questions = [
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


# ============================================================
# 1. Extract curriculum text
# ============================================================

full_text = extract_text_from_pdf(pdf_path)


# ============================================================
# 2. Load embedding model
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(embedding_model_name)

print("Embedding model loaded.")


# ============================================================
# 3. Run grid search
# ============================================================

results = []


for chunk_size in chunk_sizes:

    for overlap in overlaps:

        if overlap >= chunk_size:
            continue


        # ----------------------------------------------------
        # Create chunks
        # ----------------------------------------------------

        chunks = split_into_chunks(
            full_text,
            chunk_size=chunk_size,
            overlap=overlap
        )


        # ----------------------------------------------------
        # Create passage embeddings
        # ----------------------------------------------------

        passages = [
            f"passage: {chunk}"
            for chunk in chunks
        ]

        chunk_embeddings = model.encode(
            passages,
            normalize_embeddings=True
        )


        # ----------------------------------------------------
        # Create FAISS index
        # ----------------------------------------------------

        dimension = chunk_embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(chunk_embeddings)


        # ----------------------------------------------------
        # Print configuration
        # ----------------------------------------------------

        print()
        print("=" * 90)

        print(
            f"CHUNK SIZE={chunk_size}, "
            f"OVERLAP={overlap}, "
            f"TOTAL CHUNKS={len(chunks)}"
        )

        print("=" * 90)


        config_result = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "question_results": [],
        }


        # ====================================================
        # 4. Test each complex English question
        # ====================================================

        for test in test_questions:

            question_id = test["id"]
            question = test["question"]
            targets = test["targets"]


            # ------------------------------------------------
            # Create question embedding
            # ------------------------------------------------

            question_embedding = model.encode(
                [f"query: {question}"],
                normalize_embeddings=True
            )


            # ------------------------------------------------
            # Search all chunks
            # ------------------------------------------------

            scores, indices = index.search(
                question_embedding,
                len(chunks)
            )


            print()
            print(f"{question_id}: {question}")


            target_ranks = []


            # ------------------------------------------------
            # Check every required evidence target
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Calculate coverage and mean target rank
            # ------------------------------------------------

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


            config_result["question_results"].append(
                {
                    "id": question_id,
                    "ranks": target_ranks,
                    "top3_hits": top3_hits,
                    "top5_hits": top5_hits,
                    "mean_target_rank": mean_target_rank,
                    "total_targets": total_targets,
                }
            )


        # Save result for this chunk/overlap configuration
        results.append(config_result)


# ============================================================
# 5. Print summary table
# ============================================================

print()
print("=" * 120)

print("COMPLEX ENGLISH QUESTION GRID SEARCH SUMMARY")

print("=" * 120)


print(
    f"{'Chunk':>6} "
    f"{'Overlap':>8} "
    f"{'Q6_EN Top3':>12} "
    f"{'Q7_EN Top3':>12} "
    f"{'Q8_EN Top3':>12} "
    f"{'Total Top3':>12} "
    f"{'Total Top5':>12} "
    f"{'Mean Rank':>10}"
)

print("-" * 120)


for result in results:

    question_results = result["question_results"]


    top3_total = sum(
        question["top3_hits"]
        for question in question_results
    )


    top5_total = sum(
        question["top5_hits"]
        for question in question_results
    )


    total_targets = sum(
        question["total_targets"]
        for question in question_results
    )


    all_target_ranks = []

    for question in question_results:
        for rank in question["ranks"]:
            if rank is not None:
                all_target_ranks.append(rank)


    overall_mean_target_rank = (
        sum(all_target_ranks) / len(all_target_ranks)
        if all_target_ranks
        else None
    )


    q6 = (
        f"{question_results[0]['top3_hits']}/"
        f"{question_results[0]['total_targets']}"
    )

    q7 = (
        f"{question_results[1]['top3_hits']}/"
        f"{question_results[1]['total_targets']}"
    )

    q8 = (
        f"{question_results[2]['top3_hits']}/"
        f"{question_results[2]['total_targets']}"
    )


    print(
        f"{result['chunk_size']:>6} "
        f"{result['overlap']:>8} "
        f"{q6:>12} "
        f"{q7:>12} "
        f"{q8:>12} "
        f"{str(top3_total) + '/' + str(total_targets):>12} "
        f"{str(top5_total) + '/' + str(total_targets):>12} "
        f"{overall_mean_target_rank:>10.2f}"
    )