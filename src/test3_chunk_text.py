from pathlib import Path
from pypdf import PdfReader


pdf_path = Path(
    r"D:\Offline_AI_Teacher_Assistant\curriculum\student\Grade 7 - Science - Student Handbook - Chapter 3.pdf"
)

#previously using characters count to chunk, but found out its better to use words count instead
chunk_size = 150
overlap = 30


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


full_text = extract_text_from_pdf(pdf_path)

chunks = split_into_chunks(
    full_text,
    chunk_size=chunk_size,
    overlap=overlap
)

print(f"Total characters: {len(full_text)}")
print(f"Chunk size: {chunk_size}")
print(f"Overlap: {overlap}")
print(f"Total chunks: {len(chunks)}")

print()
print("=" * 80)

for i, chunk in enumerate(chunks[:5], start=1):
    print(f"CHUNK {i}")
    print("-" * 80)
    print(chunk)
    print()
    print("=" * 80)