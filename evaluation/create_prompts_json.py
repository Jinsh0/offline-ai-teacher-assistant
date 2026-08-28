import json
from pathlib import Path


prompts = [
    {
        "id": "Q1_EN",
        "question": "Q1",
        "title": "Practical Knowledge",
        "language": "English",
        "purpose": "Detailed factual knowledge and instruction following",
        "prompt": """I have an oven, an air fryer, and a frying pan, but no outdoor grill.

Explain how to cook a 2-3 cm thick beef steak using the reverse-sear method.

Please include:

- Preparation and seasoning
- Recommended oven temperature
- When to remove the steak from the oven
- How to sear it without burning the exterior
- Target internal temperatures for different levels of doneness
- Resting time
- A simple pan sauce
- Practical ways to reduce excessive charring, smoke, and the formation of potentially harmful compounds

Clearly distinguish established food-safety guidance from personal preference.

Keep the answer practical and well organised."""
    },

    {
        "id": "Q1_ID",
        "question": "Q1",
        "title": "Practical Knowledge",
        "language": "Bahasa Indonesia",
        "purpose": "Detailed factual knowledge and instruction following",
        "prompt": """Saya memiliki oven, air fryer, dan wajan, tetapi tidak memiliki grill untuk memasak di luar ruangan.

Jelaskan cara memasak steak sapi dengan ketebalan sekitar 2-3 cm menggunakan metode reverse-sear.

Mohon jelaskan:

- Persiapan dan pemberian bumbu
- Suhu oven yang disarankan
- Kapan steak sebaiknya dikeluarkan dari oven
- Cara melakukan searing tanpa membakar bagian luar steak
- Target suhu internal untuk berbagai tingkat kematangan
- Waktu untuk mengistirahatkan steak setelah dimasak
- Cara membuat saus sederhana menggunakan wajan
- Cara praktis untuk mengurangi gosong berlebihan, asap, dan pembentukan senyawa yang berpotensi berbahaya

Bedakan dengan jelas antara panduan keamanan pangan yang sudah memiliki dasar yang kuat dengan hal-hal yang lebih merupakan preferensi pribadi.

Buat jawabannya praktis dan terstruktur dengan baik."""
    },

    {
        "id": "Q2_EN",
        "question": "Q2",
        "title": "Comparative Reasoning",
        "language": "English",
        "purpose": "Ability to analyse trade-offs",
        "prompt": """Explain the differences between running an AI language model locally on a computer and using a cloud-based AI service.

Compare them in terms of internet dependency, privacy, cost, performance, hardware requirements, model quality and maintenance.

Then recommend which approach would be more suitable for a small rural school with unreliable internet access and limited computing equipment.

Explain the trade-offs rather than presenting one option as perfect."""
    },

    {
        "id": "Q2_ID",
        "question": "Q2",
        "title": "Comparative Reasoning",
        "language": "Bahasa Indonesia",
        "purpose": "Ability to analyse trade-offs",
        "prompt": """Jelaskan perbedaan antara menjalankan model AI secara lokal di komputer sendiri dengan menggunakan layanan AI berbasis cloud.

Bandingkan keduanya dari sisi ketergantungan terhadap koneksi internet, privasi, biaya, performa, kebutuhan perangkat keras, kualitas model, dan pemeliharaan.

Kemudian, berikan rekomendasi mengenai pendekatan mana yang lebih cocok untuk sekolah kecil di daerah pedesaan dengan akses internet yang tidak stabil dan perangkat komputer yang terbatas.

Jelaskan pro dan kontra masing-masing pendekatan, dan jangan menggambarkan salah satu pilihan sebagai solusi yang sempurna."""
    },

    {
        "id": "Q3_EN",
        "question": "Q3",
        "title": "Teacher-Oriented Task",
        "language": "English",
        "purpose": "Teacher-oriented educational use case",
        "prompt": """You are helping a Grade 7 science teacher prepare material about photosynthesis.

First, explain photosynthesis in language appropriate for a Grade 7 student.

Then create five multiple-choice questions with four options each, followed by an answer key.

Include at least:

- one question testing factual recall;
- two questions testing conceptual understanding; and
- one question requiring application of the concept to a simple real-world situation.

Keep the explanation scientifically accurate and age-appropriate."""
    },

    {
        "id": "Q3_ID",
        "question": "Q3",
        "title": "Teacher-Oriented Task",
        "language": "Bahasa Indonesia",
        "purpose": "Teacher-oriented educational use case",
        "prompt": """Anda sedang membantu seorang guru IPA (sains, ilmu pengetahuan alam) kelas 7 menyiapkan materi tentang fotosintesis.

Pertama, jelaskan fotosintesis dengan bahasa yang sesuai untuk siswa kelas 7.

Kemudian, buatlah lima soal pilihan ganda dengan masing-masing empat pilihan jawaban, disertai kunci jawaban.

Soal-soal tersebut setidaknya harus mencakup:

- satu soal yang menguji ingatan terhadap fakta;
- dua soal yang menguji pemahaman konsep; dan
- satu soal yang meminta siswa menerapkan konsep tersebut pada situasi sederhana dalam kehidupan sehari-hari.

Pastikan penjelasannya akurat secara ilmiah dan sesuai dengan tingkat usia siswa."""
    },

    {
        "id": "Q4_EN",
        "question": "Q4",
        "title": "Objectively Verifiable Reasoning",
        "language": "English",
        "purpose": "Reasoning with a clearly verifiable correct result",
        "prompt": """A school has 240 students. Three-fifths of the students participate in at least one extracurricular activity. Of those students, 25% participate in the science club.

1. How many students participate in at least one extracurricular activity?
2. How many of those students participate in the science club?

Explain the calculation step by step in a way appropriate for a Grade 7 student."""
    },

    {
        "id": "Q4_ID",
        "question": "Q4",
        "title": "Objectively Verifiable Reasoning",
        "language": "Bahasa Indonesia",
        "purpose": "Reasoning with a clearly verifiable correct result",
        "prompt": """Sebuah sekolah memiliki 240 siswa. Tiga per lima dari seluruh siswa mengikuti setidaknya satu kegiatan ekstrakurikuler. Dari siswa yang mengikuti kegiatan ekstrakurikuler tersebut, 25% mengikuti klub sains.

1. Berapa banyak siswa yang mengikuti setidaknya satu kegiatan ekstrakurikuler?
2. Berapa banyak dari siswa tersebut yang mengikuti klub sains?

Jelaskan perhitungannya langkah demi langkah dengan cara yang sesuai untuk siswa kelas 7."""
    }
]


output_file = Path("evaluation/prompts.json")

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(
        prompts,
        file,
        ensure_ascii=False,
        indent=2
    )

print(f"Created: {output_file}")