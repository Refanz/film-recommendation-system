# Laporan Proyek Machine Learning - Refanda Surya Saputra

## Project Overview

 Sebagai manusia, kita tidak bisa lepas dengan yang namanya kesibukan. Keadaan yang bisa datang kapan saja dan berasal dari mana saja, contohnya dari pekerjaan, tugas sekolah/kuliah dan hal-hal lainnya yang dapat membuat stres dan emosi. Oleh karena itu, perlu untuk istirahat sejenak agar pikiran bisa menjadi lebih tenang. Salah satu solusinya yaitu dengan menonton film favorit. Era teknologi sekarang ini, selain datang ke bioskop, banyak sekali alternatif yang dapat kita gunakan untuk menonton film, yaitu melalui smartphone kesayangan kita dengan menggunakan aplikasi seperti Youtube, Netflix atau WeTV. 

Jika kita adalah pengguna setia aplikasi streaming film seperti Netflix, di menu utama aplikasi, kita  akan disuguhkan dengan rekomendasi film yang ceritanya sesuai dengan yang biasa kita tonton. Jika kita biasanya suka menonton film komedi, maka di tampilan atas akan terlihat rekomendasi film yang memiliki genre komedi. Hal ini terjadi karena aplikasi tersebut menerapkan sistem rekomendasi yang  menggunakan algoritma atau teknik tertentu untuk memberikan saran film yang relevan dengan pengguna. 

Sistem rekomendasi akan membantu untuk menemukan film yang sesuai dengan preferensi pengguna, meningkatkan pengalaman menonton dan meningkatkan kepuasan pengguna. Manfaat yang diberikan dengan penerapan sistem rekomendasi ini dapat membuat perusahaan seperti Netflix untuk meningkatkan jumlah penggunanya. Selain itu, dengan adanya sistem rekomendasi ini kita juga dapat lebih cepat memilih film yang cocok dengan kita tanpa perlu melakukan pencarian atau scrolling film, yang mana ini memakan waktu istirahat kita. Jika tidak ada sistem rekomendasi ini, kemungkinan kita tidak akan jadi istirahat dan justru sibuk memilih film yang tepat.

Untuk menerapkan sistem rekomendasi ini terdapat beberapa metode yang digunakan untuk membuat sistem rekomendasi yaitu Content-based filtering, Collaborative-filtering, Hybrid-filtering dan metode lainnya. 

**Daftar Referensi**


**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding

Pada bagian ini, Anda perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah:
- Pernyataan Masalah 1
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Approach” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution approach (algoritma atau pendekatan sistem rekomendasi).

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.
