#%% md
# # **Proyek Sistem Rekomendasi: Rekomendasi Film Tahun 1996 - 2018**
#%% md
# ## **Import Library**
#%%
# Melakukan import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
#%% md
# **Insight:**
# 
# - Pandas: untuk manipulasi dan analisis data
# - Numpy: untuk komputasi numerik
# - Matplotlib: untuk visualisasi data
# - Tensorflow: untuk membuat model deep learning
# - Sklearn: untuk machine learning
#%% md
# ## **Loading Dataset**
# 
# Melakukan loading dataset menggunakan pandas
#%%
links_df = pd.read_csv("data/links.csv")
movies_df = pd.read_csv("data/movies.csv")
ratings_df = pd.read_csv("data/ratings.csv")
tags_df = pd.read_csv("data/tags.csv")
#%%
links_df.head(5)
#%%
movies_df.head(5)
#%%
ratings_df.head(5)
#%%
tags_df.head(5)
#%% md
# **Insight:**
# 
# - Pada proyek ini terdapat empat dataset yaitu links, movies, ratings dan tags
# - Dataset ratings berisi baris data yang mewakili satu peringkat, satu movie oleh satu pengguna. Kolom pada dataset ini adalah userId, movieId, rating dan timestamp
# - Dataset links berisi data movie dengan tautan ke sumber data film yaitu IMDB dan TMDB. Kolom pada dataset ini adalah movieId, imdbId dan tmdbId
# - Dataset movies berisi informasi dari film yaitu judul dan genre. Kolom pada dataset ini adalah movieId, title dan genres
# - Dataset tags berisi informasi tag yang diberikan oleh satu user pada satu film. Kolom pada dataset ini adalah userId, movieId, tag dan timestamp
# 
#%% md
# ## **Exploratory Data Analysis (EDA)**
#%% md
# ### **Melihat Informasi Dataset**
#%%
movies_df.info()
#%% md
# **Insight:**
# 
# Berdasarkan output di atas, dataset movies memiliki 7742 entri. Terdapat tiga variabel pada dataset ini yaitu movieId, title atau judul film dan genres.
# 
#%%
ratings_df.info()
#%% md
# **Insight:**
# 
# Berdasarkan output di atas, dataset ratings memiliki 100835 entri. Terdapat empat variabel pada dataset ini yaitu userId, movieId, rating dan timestamp.
#%%
links_df.info()
#%% md
# **Insight:**
# 
# Berdasarkan output di atas, dataset links memiliki 9741 entri. Terdapat tiga variabel pada dataset ini yaitu movieId, imdbId dan tmdbId.
#%%
tags_df.info()
#%% md
# **Insight:**
# 
# Berdasarkan output di atasm dataset tags memiliki 3682 entri. Terdapat empat variabel pada dataset ini yaitu userId, movieId, tag dan timestamp.
#%%
print('Jumlah userId: ', len(ratings_df['userId'].unique()))
print('Jumlah data film: ', len(movies_df['movieId']))
print('Jumlah data rating: ', len(ratings_df['rating']))
#%% md
# **Insight:**
# 
# Terdapat 610 user yang memberikan rating pada 9742 film. Kemudian untuk akumulasi data rating yang diberikan user totalnya adalah 100836.
#%% md
# ### **Mengecek Missing Value**
#%%
links_df.isna().sum()
#%%
ratings_df.isna().sum()
#%%
movies_df.isna().sum()
#%%
tags_df.isna().sum()
#%% md
# **Insight:**
# 
# Dari keempat dataset di atas, setelah mengecek menggunakan isna() tidak ditemukan adanya missing value/Nan.
#%% md
# ### **Melihat Deskripsi Statistik pada Dataset Rating**
#%%
ratings_df.describe()
#%% md
# **Insight:**
# 
# Dari output di atas, ditemukan bahwa nilai minimum user memberikan rating adalah 0.5 dan maksimum ratingnya adalah 5. Ini artinya, skala rating berkisar antara 0.5 hingga 5. Kemudian untuk rata-rata user memberikan rating 3.5.
#%% md
# ### **Mengecek Nilai Duplikat pada Dataset**
#%%
print("Jumlah data duplikat pada links.csv: ", links_df.duplicated().sum())
print("Jumlah data duplikat pada movies.csv: ", movies_df.duplicated().sum())
print("Jumlah data duplikat pada ratings.csv: ", ratings_df.duplicated().sum())
print("Jumlah data duplikat pada tags.csv: ", tags_df.duplicated().sum())
#%% md
# **Insight:**
# 
# Dari keempat dataset di atas, setelah melakukan pengecekan dengan fungsi duplicated() tidak ditemukan adanya data yang sama/duplikat.
#%% md
# ## **Data Preprocessing**
# 
# Pada tahapan ini akan menggabungkan beberapa dataset untuk menjadi dataset utama yang digunakan untuk modeling
#%%
# Menggabungkan dataset movies dengan ratings
main_movies_df = pd.merge(movies_df, ratings_df[['movieId', 'rating', 'userId']], on='movieId', how='left')

main_movies_df
#%%
main_movies_df.isna().sum()
#%%
print("Jumlah data duplikasi: ", main_movies_df.duplicated().sum())
#%% md
# **Insight:**
# 
# - Dari tahapan ini dihasilkan dataframe dari hasil gabungan dataset movies dan ratings. main_movies_df akan digunakan untuk membuat sistem rekomendasi film.
# - Setelah dilakukan merge movies dan ratings, kemudian melakukan pengecekan missing value dengan isna(). Ditemukan untuk variabel rating dan userId terdapat missing value dengan jumlah masing-masing adalah 18 data. Masalah ini akan ditangani pada tahapan Data Preparation.
# - Setelah merging tidak ditemukan data duplikasi
# 
#%% md
# ## **Data Preparation**
#%% md
# ### **Mengatasi Missing Value**
#%%
# Memberishkan missing value dengan fungsi dropna()
main_movies_clean_df = main_movies_df.dropna()
main_movies_clean_df
#%%
# Mengecek kembali missing value pada variabel main_movies_clean_df
main_movies_clean_df.isnull().sum()
#%% md
# **Insight:**
# Langkah pertama pada tahapan Data Preparation adalah memersihkan missing value pada main_movies_df menggunakan fungsi dropna(). Kemudian dilakukan pengecekan kembali untuk memastikan apakah masalah missing value sudah ditangani. Terlihat pada output di atas sudah tidak ada missing value dan siap untuk lanjut ke proses selanjutnya.
# 
#%% md
# ### **Konversi Data Series menjadi List**
#%%
# Mengkonversi data series (movieId, title dan genres) menjadi dalam bentuk list
movie_id = main_movies_clean_df['movieId'].tolist()
movie_title = main_movies_clean_df['title'].tolist()
movie_genre = main_movies_clean_df['genres'].tolist()

print(len(movie_id))
print(len(movie_title))
print(len(movie_genre))
#%%
# Membuat dictionary untuk data (movieId, title dan genres)
movie_new = pd.DataFrame({
    'id': movie_id,
    'title': movie_title,
    'genres': movie_genre,
}).drop_duplicates()

movie_new
#%% md
# **Insight:**
# 
# Mengambil tiga variabel pada main_movies_clean_df yaitu movieId, title dan genre. Kemudian dari ketiga variabel ini diubah menjadi list. Setelah itu, membuat dictionary dengan pasangan key:value sesuai dengan variabel yang diambil. Kemudian dibuat dataframe yaitu movie_new dari dictionary yang sudah dibuat.
#%% md
# ### **Melakukan Formatting pada Variabel Genres**
#%%
movies_formatted_df = movie_new.iloc[:]
movies_formatted_df['genres'] = movies_formatted_df['genres'].str.replace('-', '', regex=False)
movies_formatted_df
#%% md
# **Insight:**
# 
# Terdapat nilai pada variabel genres yang dipisahkan dengan tanda "-". Untuk memudahkan proses vektorisasi nantinya, maka perlu mengghapus tanda "-" menjadi tanpa spasi.
# 
#%% md
# ### **Menghapus Data dengan Genre yang Tidak Jelas**
#%%
fix_movies_df = movies_formatted_df.iloc[:]

fix_movies_df['genres'].unique()
#%%
fix_movies_df[fix_movies_df['genres'] == '(no genres listed)']
#%%
fix_movies_df = fix_movies_df.replace('(no genres listed)', np.nan)
fix_movies_df = fix_movies_df.dropna()

fix_movies_df
#%% md
# **Insight:**
# 
# Terdapat film dengan genre yang tidak jelas yaitu "(no genres listed)". Untuk menjaga kualitas data, maka perlu dilakukan penghapusan pada data tersebut.
#%% md
# ### **Melakukan Encoding Fitur userId dan movieId ke dalam Indeks Integer**
#%%
# Membuat dataframe baru
new_rating_df = ratings_df.drop(columns=['timestamp'])

# Mengubah userId menjadi list tanpa nilai yang sama
user_ids = new_rating_df['userId'].unique().tolist()
print('list userId: ', user_ids[:10])

# Melakukan encoding userId
user_to_user_encoding = {
    x: i for i, x in enumerate(user_ids)
}
print('encoded userId [1]: ', user_to_user_encoding[1])

# Melakukan proses encoding angka ke userId
user_encoded_to_user = {
    i: x for i, x in enumerate(user_ids)
}
print('encoding angka ke userId [0]: ', user_encoded_to_user[0])
#%%
# Mengubah movieId menjadi list tanpa nilai yang sama
movie_ids = new_rating_df['movieId'].unique().tolist()
print('list movieId: ', movie_ids[:10])

# Melakukan proses encoding movieId
movie_to_movie_encoded = {
    x: i for i, x in enumerate(movie_ids)
}
print('encoded movieId [1]: ', movie_to_movie_encoded[1])

# Melakukan proses encoding angka ke movieId
movie_encoded_to_movie = {
    i: x for i, x in enumerate(movie_ids)
}
print('encoding angka ke movieId [0]: ', movie_encoded_to_movie[0])
#%%
# Mapping userId ke dataframe user
new_rating_df['user'] = new_rating_df['userId'].map(user_to_user_encoding)

# Mapping movieId ke dataframe movie
new_rating_df['movie'] = new_rating_df['movieId'].map(movie_to_movie_encoded)
#%% md
# **Insight:**
# 
# Tahapan ini digunakan untuk mengubah ID asli (userId dan movieId) yang mungkin tidak berurutan menjadi ID baru yang berurutan mulai dari 0, 1, 2, dan seterusnya.
#%% md
# ### **Mempersiapkan Data untuk Modeling Collaborative Filtering**
#%%
# Mendaoatkan jumlah user
num_users = len(user_to_user_encoding)

# Mendapatkan jumlah film
num_movie = len(movie_encoded_to_movie)

# Mendapatkan minimum dan maksimum rating
min_rating = min(new_rating_df['rating'])
max_rating = max(new_rating_df['rating'])

print("Number of User: {}\nNumber of Movie: {}\nMin Rating: {}\nMax Rating: {}".format(num_users, num_movie, min_rating,
                                                                                       max_rating))
#%% md
# **Insight:**
# 
# Mempersiapkan data yaitu jumlah user, jumlah film, nilai minimum rating dan nilai maksimu rating. Data ini digunakan untuk modeling dengan pendekatan Collaborative Filtering.
# 
#%% md
# ### **Membagi Data untuk Training dan Validasi**
#%%
# Mengacak data
new_rating_df = new_rating_df.sample(frac=1, random_state=42)
new_rating_df
#%% md
# **Insight:**
# 
# Sebelum melakukan spliting dataset, data diacak terelbih dahulu supaya distribusinya menjadi random.
#%%
# Membuat variabel x untuk mencocokkan data user dan movie menjadi satu value
x = new_rating_df[['user', 'movie']].values

# Membuat variabel y untuk membuat rating dari hasil
y = new_rating_df['rating'].apply(lambda z: (z - min_rating) / (max_rating - min_rating)).values

# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * new_rating_df.shape[0])

x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],
)

print(x, y)
#%% md
# **Insight:**
# 
# Melakukan spliting dataset menjadi data train dan data validasi dengan perbandingan 80:20. Sebelum melakukan spliting dataset, dilakukan scaling nilai variabel rating dalam rentang 0 sampai 1 untuk mempermudah proses training. Data train dan validasi ini digunakan untuk modeling dengan pendekatan Collaborative Filtering.
#%% md
# ## **Modeling**
#%% md
# ### **Model Development dengan Content Based Filtering**
# 
# Mengembangkan sistem rekomendasi film dengan pendekatan content based filtering berdasarkan genre film
#%% md
# #### **Mempersiapkan data untuk modeling CBF**
#%%
data_cbf = fix_movies_df.iloc[:]
data_cbf.sample(5)
#%% md
# #### **Menemukan Representasi Fitur Penting**
#%%
# Inisialisasi TfidVectorizer
tfid = TfidfVectorizer()

# Melakukan perhitungan idf pada data genres
tfid.fit(data_cbf['genres'])

# Mapping array dari fitur index integer ke fitur nama
tfid.get_feature_names_out()
#%% md
# **Insight:**
# 
# Dari hasil vektorisasi menggunakan TfidfVectorizer didapatkan fitur yaitu
# ['action', 'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'filmnoir', 'horror', 'imax', 'musical', 'mystery', 'romance', 'scifi', 'thriller', 'war', 'western']
# 
#%%
# Melakukukan fit lalu ditransformasikan ke bentuk matrix
tfidf_matrix = tfid.transform(data_cbf['genres'])

# Melihat ukuran matrix tfidf
tfidf_matrix.shape
#%% md
# **Insight:**
# 
# Hasil matrix memiliki ukuran (9690, 19). Nilai 9690 merupakan ukuran data dan 19 merupakan matrix genre film.
#%%
# Mengubah vektor tfidf dalam bentuk matriks dengan fungsi todense()
tfidf_matrix.todense()
#%%
# Membuat dataframe untuk melihat tfidf matrix
# Kolom diisi dengan genre film
# Baris diisi dengan judul film

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tfid.get_feature_names_out(),
    index=data_cbf['title'],
)
#%% md
# **Insight:**
# 
# Dari hasil moutput matriks tfid di atas menunjukkan film Toy Story (1995) memiliki genre adventure, animation, children, comedy dan fantasy. Hal ini terlihat dari nilai matriks 0.416775 pada genre adventure, 0.516403 pada genre animation, 0.504783 pada genre children, 0.267318 pada genre comedy dan 0.483075 pada genre fantasy.
# 
#%% md
# #### **Cosine Similarity**
#%%
# Menghitung cosine similarity pada matrix tfidf
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim
#%% md
# **Insight:**
# 
# Pada tahap ini dilakukan perhitungan cosine similarity pada dataframe tfidf_matrix. Menggunakan fungsi consine_similarity dari library sklearn. Terlihat pada output di atas adalah matriks kesamaan dalam bentuk array.
#%%
# Membuat dataframe dari variabel cosine_sim dengan baris dan kolo berupa nama film
cosine_sim_df = pd.DataFrame(cosine_sim, index=data_cbf['title'], columns=data_cbf['title'])
print("Shape: ", cosine_sim_df.shape)

# Melihat similarity matrix pada setiap film
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)
#%% md
# **Insight:**
# 
# Shape (9690, 9690) merupakan ukuran matriks similarity dari data film. Dari output di atas, sebagai contoh film Taken (2008) memiliki indikasi kesamaan dengan film Cleanskin (2012) dengan nilai similarity adalah 0.857362 dan Knight's Tale, A (2001) nilainya 0.316767.
#%% md
# #### **Mendapatkan Rekomendasi Film**
#%%
def movie_recommendations(movie_title, similarity_data=cosine_sim_df, items=data_cbf[['title', 'genres']], k=5):
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    index = similarity_data.loc[:, movie_title].to_numpy().argpartition(range(-1, -k, -1))

    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k + 2): -1]]

    # Drop movie_title agar judul film yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(movie_title, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)
#%% md
# **Insight:**
# 
# Fungsi ini bertujuan untuk memberikan rekomendasi film berdasarkan judul film yang diberikan, dengan memanfaatkan data kemiripan (similarity) yang sudah dihitung sebelumnya.
# 
# Secara ringkas, alur kerjanya adalah sebagai berikut:
# 
# 1. Mencari Indeks Film Termirip: Baris pertama menggunakan .argpartition() untuk cara yang sangat cepat dalam menemukan posisi (indeks) dari k film yang paling mirip dengan movie_title tanpa perlu mengurutkan seluruh data.
# 2. Mengambil Judul Film: Berdasarkan posisi tersebut, baris kedua mengambil judul-judul film yang paling mirip dan mengurutkannya dari yang paling tinggi kemiripannya.
# 3. Menghapus Film Asli: Baris ketiga memakai .drop() untuk menghapus movie_title dari daftar rekomendasi, agar film tersebut tidak merekomendasikan dirinya sendiri.
# 4. Menampilkan Hasil Akhir: Terakhir, fungsi ini menggabungkan daftar judul film yang sudah bersih dengan data items (yang berisi judul dan genre), lalu mengembalikan k rekomendasi teratas dalam bentuk tabel (DataFrame) yang rapi.
#%%
# Variabel untuk menyimpan top-N film yang direkoendasikan
k = 5

# Mendapatkan rekomendasi film yang mirip dengan Toy Story (1995)
target = 'Toy Story (1995)'
results = movie_recommendations(target, k=k)
results
#%% md
# **Insight:**
# 
# Melakukan percobaan untuk mencari rekomendasi film yang sama dengan film Toy Story (1995). Didapatkan hasil film yang mirip berdasarkan genrenya adalah Moana (2016), Tale of Despereaux, The (2008), Shrek the Third (2007), Wild, The (2006), Adventures of Rocky and Bullwinkle, The (2000), Emperor's New Groove, The (2000)
#%% md
# #### **Evaluasi Hasil**
#%%
def search_movie(*targets):
    founded_movie = data_cbf[data_cbf['title'].isin(targets)]

    return {
        'title': founded_movie['title'].tolist()[0],
        'genres': founded_movie['genres'].tolist()[0],
    }


def evaluating_result(target, recommendation_results, k):
    movie_target = search_movie(target)
    result_genre = recommendation_results['genres']

    count = 0

    for mov in result_genre:
        if mov in movie_target['genres']:
            count += 1

    precision_k = count / k

    eval_result = {
        "movie_title": movie_target['title'],
        "genres": movie_target['genres'],
        "precision@k": precision_k,
        "recommendation_results": recommendation_results,
    }

    return eval_result


print("Hasil Evaluasi: \n")
eval_result = evaluating_result(target, results, k)
for key in evaluating_result(target, results, k):
    print(f"{key}: {eval_result[key]}")
#%% md
# **Insight:**
# 
# - Melakukan evaluasi sederhana terhadap hasil dari sistem rekomendasi film dengan pendekatan Content-based Filtering berdasarkan genre film. Secara spesifik, kode ini menghitung metrik Precision@k dengan cara memeriksa kesamaan genre antara film target dan film-film yang direkomendasikan.
# - Hasil evaluasi ini menunjukkan performa yang sempurna dengan skor precision@k sebesar 1.0 (atau 100%) untuk film target "Toy Story (1995)". Skor maksimal ini tercapai karena kelima film yang direkomendasikan oleh sistem (dari "Moana" hingga "The Adventures of Rocky and Bullwinkle") memiliki kombinasi genre yang identik dengan film target. Artinya, berdasarkan logika evaluasi yang memeriksa kesamaan genre, setiap rekomendasi dianggap sebagai "cocok" atau relevan, membuktikan bahwa untuk kasus ini, sistem sangat efektif dalam menemukan film-film dengan klasifikasi konten yang sama persis.
#%% md
# ### **Model Development dengan Collaborative Filtering**
#%% md
# #### **Training Model**
#%%
class RecommenderNet(keras.Model):

    # Inisialisasi fungsi
    def __init__(self, num_users, num_movie, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movie = num_movie
        self.embedding_size = embedding_size

        # layer Embedding user
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )

        # layer embedding user bias
        self.user_bias = layers.Embedding(num_users, 1)

        # layer embeddings movie
        self.movie_embedding = layers.Embedding(
            num_movie,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )

        # layer embedding movie bias
        self.movie_bias = layers.Embedding(num_movie, 1)

    def call(self, inputs):
        # memanggil layer embedding 1
        user_vector = self.user_embedding(inputs[:, 0])
        # memanggil layer embedding 2
        user_bias = self.user_bias(inputs[:, 0])
        # memanggil layer embedding 3
        movie_vector = self.movie_embedding(inputs[:, 1])
        # memanggil layer embedding 4
        movie_bias = self.movie_bias(inputs[:, 1])

        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)

        x = dot_user_movie + user_bias + movie_bias

        # Aktivasi sigmoid
        return tf.nn.sigmoid(x)
#%% md
# **Insight:**
# 
# Kode di atas membuat sebuah model neural network untuk sistem rekomendasi film dengan pendekatan Collaborative Filtering. Secara spesifik, ini adalah implementasi dari teknik Matrix Factorization menggunakan Keras.
# 
#%%
# Inisialisasi model
model = RecommenderNet(num_users, num_movie, 30)

# Compile model
model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[keras.metrics.RootMeanSquaredError()],
)
#%% md
# **Insight:**
# 
# Model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam sebagai optimizer dan root mean squared erro (RMSE) sebagai metrik evaluasi.
#%%
# Memulai training
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=8,
    epochs=50,
    validation_data=(x_val, y_val)
)
#%% md
# **Insight:**
# 
# Kode ini adalah perintah untuk memulai proses training (pelatihan) model dengan konfigurasi sebagai berikut:
# - x = x_train, y = y_train: Model akan belajar dari data training (x_train) untuk bisa menebak y_train dengan benar.
# - batch_size = 8: Data akan diproses dalam kelompok-kelompok kecil berisi 8 sampel untuk efisiensi.
# - epochs = 50: Seluruh proses belajar ini akan diulang sebanyak 50 kali.
# - validation_data = (x_val, y_val): Setelah setiap putaran (epoch), performa model akan dievaluasi menggunakan data validasi untuk memantau apakah terjadi overfitting.
# 
# Hasil dari proses training ini (seperti riwayat loss dan RMSE) akan disimpan dalam variabel history.
#%% md
# #### **Visualisasi Metrik**
#%%
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('Root Mean Squared Error')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#%% md
# **Insight:**
# 
# Dari hasil grafik di atas, setelah proses training model didapatkan nilai error akhir sebesar 0.1732 dan error pada data validasi sebesar 0.2007. 
# 
#%% md
# #### **Mendapatkan Rekomendasi Film**
#%%
df_movies = fix_movies_df.iloc[:]
movie = pd.read_csv('data/ratings.csv')

# Mengambil sampel user
user_id = movie['userId'].sample(1).iloc[0]
movie_watched_by_user = movie[movie['userId'] == user_id]

movie_not_watched = df_movies[~df_movies['id'].isin(movie_watched_by_user['movieId'].values)]['id']
movie_not_watched = list(
    set(movie_not_watched)
    .intersection(
        set(movie_to_movie_encoded.keys())
    )
)

movie_not_watched = [[movie_to_movie_encoded.get(x)] for x in movie_not_watched]
user_encoder = user_to_user_encoding.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movie_not_watched), movie_not_watched)
)
#%% md
# **Insight:**
# 
# Kode ini bertujuan untuk mempersiapkan data input yang akan diberikan ke model rekomendasi untuk mendapatkan prediksi bagi satu pengguna spesifik.  Hasil akhirnya adalah data yang siap dimasukkan ke model.predict() untuk memperkirakan rating semua film yang belum ditonton oleh pengguna tersebut.
#%%
ratings = model.predict(user_movie_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [
    movie_encoded_to_movie.get(movie_not_watched[x][0]) for x in top_ratings_indices
]

print("Showing recommendations for users: {}".format(user_id))
print("====" * 8)
print("Movie with high ratings from user")
print('----' * 14)

top_movie_user = (
    movie_watched_by_user.sort_values(
        by='rating',
        ascending=False
    )
    .head(5)['movieId'].values
)

movie_df_rows = df_movies[df_movies['id'].isin(top_movie_user)]
for row in movie_df_rows.itertuples():
    print(row.title, ':', row.genres)

print("\n")
print('----' * 15)
print('Top 10 movie recommendation')
print('----' * 15)

recommended_movie = df_movies[df_movies['id'].isin(recommended_movie_ids)]
for row in recommended_movie.itertuples():
    print(row.title, ':', row.genres)

#%% md
# **Insight:**
# 
# - Kode ini menampilkan dua bagian informasi:
#   - Film yang Sudah Disukai Pengguna: Menampilkan 5 film dengan rating tertinggi yang pernah diberikan oleh pengguna sebagai referensi seleranya.
#   - Top 10 Rekomendasi: Menampilkan judul dan genre dari 10 film baru yang paling direkomendasikan oleh model untuk pengguna tersebut.
# - Dari hasil rekomendasi di atas untuk user dengan id 425, diperoleh rekomendasi beberapa film yang sesuai dengan rating user yaitu dengan genre Action, Drama, Comedy, War, Thriller dan Romance