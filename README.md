# Laporan Proyek Machine Learning - Refanda Surya Saputra

## Project Overview

 Sebagai manusia, kita tidak bisa lepas dengan yang namanya kesibukan. Keadaan yang bisa datang kapan saja dan berasal dari mana saja, contohnya dari pekerjaan, tugas sekolah/kuliah dan hal-hal lainnya yang dapat membuat stres dan emosi. Oleh karena itu, perlu untuk istirahat sejenak agar pikiran bisa menjadi lebih tenang. Salah satu solusinya yaitu dengan menonton film favorit. Era teknologi sekarang ini, selain datang ke bioskop, banyak sekali alternatif yang dapat kita gunakan untuk menonton film, yaitu melalui smartphone kesayangan kita dengan menggunakan aplikasi seperti Youtube, Netflix atau WeTV. 

Jika kita adalah pengguna setia aplikasi streaming film seperti Netflix, di menu utama aplikasi, kita  akan disuguhkan dengan rekomendasi film yang ceritanya sesuai dengan yang biasa kita tonton. Jika kita biasanya suka menonton film komedi, maka di tampilan atas akan terlihat rekomendasi film yang memiliki genre komedi. Hal ini terjadi karena aplikasi tersebut menerapkan sistem rekomendasi yang  menggunakan algoritma atau teknik tertentu untuk memberikan saran film yang relevan dengan pengguna. 

Sistem rekomendasi akan membantu untuk menemukan film yang sesuai dengan preferensi pengguna, meningkatkan pengalaman menonton dan meningkatkan kepuasan pengguna. Manfaat yang diberikan dengan penerapan sistem rekomendasi ini dapat membuat perusahaan seperti Netflix untuk meningkatkan jumlah penggunanya. Selain itu, dengan adanya sistem rekomendasi ini kita juga dapat lebih cepat memilih film yang cocok dengan kita tanpa perlu melakukan pencarian atau scrolling film, yang mana ini memakan waktu istirahat kita. Jika tidak ada sistem rekomendasi ini, kemungkinan kita tidak akan jadi istirahat dan justru pusing karena sibuk memilih film yang tepat.

Terdapat berbagai metode untuk menerapkan sistem rekomendasi, di antaranya adalah Content-based Filtering, Collaborative Filtering, dan Hybrid Filtering. Dari berbagai metode tersebut, proyek ini akan berfokus pada penerapan dua pendekatan, yaitu Content-based dan Collaborative Filtering. Pendekatan pertama, Content-based Filtering, berfokus pada analisis atribut atau konten dari item itu sendiri. Sebagaimana ditunjukkan oleh Siagian et al. [1], metode ini terbukti efektif memberikan rekomendasi berdasarkan deskripsi atau sinopsis film yang pernah disukai pengguna. Di sisi lain, pendekatan Collaborative Filtering bekerja dengan cara yang berbeda, yaitu memberikan rekomendasi berdasarkan interaksi dan rating pengguna tanpa perlu menganalisis konten item. Menurut Wiputra dan Shandi [2], metode ini mampu mengelompokkan item dan pengguna berdasarkan pola rating yang serupa. Kombinasi kedua metode ini memungkinkan sistem untuk memberikan rekomendasi yang tidak hanya personal, tetapi juga beragam."

**Daftar Referensi**

[1] R. I. P. Siagian, N. Khoiriah, S. A. Priscilia, M. R. A. Tanjung, and A. Perdana, "Penerapan Machine Learning untuk Rekomendasi Film Berdasarkan Preferensi Pengguna," JATI (Jurnal Mahasiswa Teknik Informatika), vol. 9, no. 4, pp. 5658-5662, 2025.

[2] M. M. Wiputra and Y. J. Shandi, "Perancangan Sistem Rekomendasi Menggunakan Metode Collaborative Filtering dengan Studi Kasus Perancangan Website Rekomendasi Film," Media Informatika, vol. 20, no. 1, pp. 1-8, 2021.


## Business Understanding

Sistem rekomendasi merupakan teknologi cerdas yang telah menjadi strategi bisnis bagi platform digital, dengan cara menganalisis data perilaku pengguna dan atribut item untuk menyajikan konten secara personal. Platform-platform terkemuka memanfaatkan teknologi ini untuk mempertahankan pengguna. Dengan secara proaktif menyajikan film yang relevan, layanan-layanan ini berhasil menekan tingkat pelanggan yang berhenti berlangganan. Hal ini membuktikan bahwa implementasi sistem rekomendasi bukanlah sekadar fitur teknis, melainkan sebuah solusi strategis untuk pertumbuhan dan keberlanjutan bisnis di era digital.


### Problem Statements

- Bagaimana cara menggunakan atribut konten film, khususnya genre, untuk merekomendasikan film lain yang memiliki kemiripan dengan film-film yang telah disukai pengguna?
- Bagaimana cara menggunakan data rating dari seluruh pengguna untuk menemukan pengguna lain dengan selera serupa, untuk merekomendasikan film yang kemungkinan besar akan disukai?

### Goals

- Menggunakan pendekatan Content-based Filtering untuk menghasilkan rekomendasi film yang memiliki genre yang serupa
- Menggunakan pendekatan Collaborative-Filtering untuk menghasilkan rekomendasi film yang belum pernah ditonton oleh pengguna

### Solution Statements

- Membuat sistem rekomendasi film dengan dua pendekatan yaitu Content-based Filtering dan Collaborative Filtering
- Menerapkan regularisasi L2 di lapisan embedding  pada pendekatan Collaborative Filtering  untuk mencegah overfitting

## Data Understanding

Data yang digunakan pada proyek sistem rekomendasi film ini adalah Movie Lens Small Latest Dataset yang diunduh dari [Kaggle](https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset). Dataset ini terdiri dari empat file yaitu links.csv, movies.csv, tags.csv dan ratings.csv. Dataset ini mengandung 100836 rating dan 3683 tag pada 9742 film yang diberikan oleh pengguna. Data ini dibuat oleh 610 pengguna antara Maret 29, 1996 dan September 24, 2018.

### Variabel-variabel pada Movie Lens Small Latest Dataset adalah sebagai berikut:

- **movies**: ini adalah data film meliputi judul dan genre
- **links**: ini adalah data sumber film yaitu tmd dan imdb
- **tags**: ini adalah data tag pada film yang diberikan oleh pengguna
- **ratings**: ini adalah rating yang diberikan pada pengguna pada film yang sudah ditonton

### Exploratory Data Analysis

Berikut ini adalah beberapa tahapan EDA yang telah dilakukan pada proyek ini.

#### Melihat Informasi Dataset

**Links Dataset**

<img src="assets/info-links.png" width=100% alt="Info Dataset" >

Berdasarkan informasi di atas, dataset links memiliki 9741 entri. Terdapat tiga variabel pada dataset ini yaitu movieId, imdbId dan tmdbId.

**Movies Dataset**

<img src="assets/info-movies.png" width=100% alt="Info Dataset" >

Berdasarkan informasi di atas, dataset movies memiliki 7742 entri. Terdapat tiga variabel pada dataset ini yaitu movieId, title atau judul film dan genres.

**Ratings Dataset**

<img src="assets/info-ratings.png" width=100% alt="Info Dataset" >

Berdasarkan informasi di atas, dataset ratings memiliki 100835 entri. Terdapat empat variabel pada dataset ini yaitu userId, movieId, rating dan timestamp.

**Tags Dataset**

<img src="assets/info-tags.png" width=100% alt="Info Dataset" >

Berdasarkan informasi di atas dataset tags memiliki 3682 entri. Terdapat empat variabel pada dataset ini yaitu userId, movieId, tag dan timestamp.

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
