from pathlib import Path
from pypdf import PdfReader

pdf_path = Path(
    r"D:\Offline_AI_Teacher_Assistant\curriculum\student\Grade 7 - Science - Student Handbook - Chapter 3.pdf"
)

reader = PdfReader(pdf_path)

print(f"Pages found: {len(reader.pages)}")

all_text = []

for i, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    all_text.append(text)

    print(f"Page {i}: {len(text)} characters extracted")

full_text = "\n".join(all_text)

print()
print(f"Total characters extracted: {len(full_text)}")
print()
print("First 1500 characters:")
print("-" * 60)
print(full_text[:1500])