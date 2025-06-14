# Laporan Proyek Machine Learning - Lilis Indra Purnama Sari

## Domain Proyek

**Latar Belakang:**

Pertanian merupakan sektor penting dalam menjaga ketahanan pangan nasional. Produksi hasil panen menjadi penopang utama dalam pemenuhan kebutuhan pangan masyarakat. Namun demikian, hasil panen sering mengalami fluktuasi dari tahun ke tahun yang disebabkan oleh berbagai faktor, seperti perubahan iklim, pola tanam, serangan hama, dan variabilitas cuaca ekstrem (NASA 2023).

Ketidakpastian ini menyulitkan petani, lembaga penyimpanan pangan, serta pengambil kebijakan dalam merencanakan distribusi, logistik, dan pengendalian harga. Oleh karena itu, dibutuhkan sistem prediksi hasil panen yang akurat berdasarkan data historis untuk mendukung proses pengambilan keputusan yang lebih baik, proaktif, dan efisien.

Menurut FAO (2023), penggunaan pendekatan berbasis data dan model prediktif dalam pertanian terbukti mampu meningkatkan efisiensi produksi serta mengurangi kerugian pascapanen. Sementara itu, laporan dari World Bank (2021) menyebutkan bahwa pemanfaatan teknologi digital dan machine learning dalam sektor pertanian dapat menjadi solusi untuk mengatasi ketidakpastian hasil panen akibat perubahan iklim.

Dengan memanfaatkan pendekatan regresi, data historis hasil panen dapat diolah untuk memprediksi produksi di masa depan. Hasil prediksi ini sangat berguna dalam perencanaan musim tanam, estimasi cadangan pangan, serta pengambilan kebijakan subsidi dan distribusi logistik secara lebih tepat sasaran.

**Referensi:**

Food and Agriculture Organization of the United Nations (FAO). (2023). Data-driven agriculture: Enhancing food security through digital innovation. Retrieved from https://www.fao.org

NASA. (2023). Climate change and agriculture. Retrieved from https://earthobservatory.nasa.gov/features/ClimateAgriculture

Shawon, S. M., Ema, F. B., & Mahi, A. K. (2024). Crop yield prediction using machine learning: An extensive and systematic literature review. Smart Agricultural Technology, 10.

World Bank. (2021). Digital agriculture profile: Indonesia. Retrieved from https://www.worldbank.org

## Business Understanding
### Problem Statements
 - Hasil panen tanaman seringkali fluktuatif dan sulit diprediksi karena dipengaruhi banyak faktor seperti cuaca, jenis tanah, dan metode budidaya.
 - Belum tersedia sistem prediksi hasil panen berbasis data historis yang memanfaatkan teknologi machine learning secara optimal.
 - Pengaruh masing-masing faktor seperti curah hujan, suhu, jenis tanah, dan penggunaan irigasi terhadap hasil panen belum dianalisis secara mendalam.
 - Perbedaan kondisi antar wilayah belum diperhitungkan secara tepat dalam model prediksi hasil panen.

### Goals
 - Mengembangkan model prediksi hasil panen menggunakan data historis cuaca (temperature, rainfall, weather condition), lingkungan (soil type, region), dan praktik pertanian (fertilizer used, irrigation used, days to harvest).
 - Menerapkan algoritma machine learning (Linear Regression, Random Forest, XGBoost) untuk membangun model prediktif yang akurat dan terukur.
 - Membandingkan performa berbagai model regresi untuk menentukan pendekatan terbaik dalam prediksi hasil panen.
 - Menganalisis fitur-fitur penting yang paling memengaruhi hasil panen dengan menggunakan feature importance dan koefisien regresi.

### Solution statements
a. Melatih model baseline menggunakan Linear Regression sebagai pembanding awal.

b. Membangun model regresi lanjutan dengan Random Forest Regressor untuk menangani non-linearitas dan fitur interaktif dengan melakukan tuning hyperparameter.

c. Membangun model regresi lanjutan kedua menggunakan XGBoost Regressor untuk mengeksplorasi performa model berbasis boosting dengan melakukan tuning hyperparameter.

d. Mengevaluasi setiap model menggunakan metrik RMSE, R square dan MAE, serta membandingkan hasilnya untuk memilih model terbaik.


## Data Understanding
- Sumber Data: Kaggle — Agricultural Crop Yield Dataset (https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield)
- Jumlah data: 1.000.000 amatan dengan 10 fitur 

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
a. Region : The geographical region where the crop is grown (North, East, South, West).

b. Soil_Type: The type of soil in which the crop is planted (Clay, Sandy, Loam, Silt, Peaty, Chalky).

c. Crop: The type of crop grown (Wheat, Rice, Maize, Barley, Soybean, Cotton).

d. Rainfall_mm: The amount of rainfall received in millimeters during the crop growth period.

e. Temperature_Celsius: The average temperature during the crop growth period, measured in degrees Celsius.

f. Fertilizer_Used: Indicates whether fertilizer was applied (True = Yes, False = No).

g. Irrigation_Used: Indicates whether irrigation was used during the crop growth period (True = Yes, False = No).

h. Weather_Condition: The predominant weather condition during the growing season (Sunny, Rainy, Cloudy).

i. Days_to_Harvest: The number of days taken for the crop to be harvested after planting.

j. Yield_tons_per_hectare: The total crop yield produced, measured in tons per hectare.

**Rubrik/Kriteria Tambahan (Exploratory Data Analysis)**:
- Memeriksa struktur data -> Terdapat 6 fitur kategorik dan terdaoat 4 fitur numerik
- Mengecek missing value -> Tidak ditemukan adanya missing value
- Mengecek data duplikat -> Tidak ditemukan adanya data duplikat
- Melihat distribusi setiap target -> Target berdistribusi normal, fitur numerik memiliki distribusi seragam, dan fitur kategorik memiliki frekuensi yang cukup seimbang
- Melihat korelasi antar peubah numerik -> Tidak ditemukan adanya indikasi multikolinearitas
- Melihat korelasi antar peubah kategorik -> Tidak ditemukan adanya indikasi multikolinearitas
- Melihat pengaruh setiap fitur kategorik terhadap target -> Penggunaan irigasi memiliki korelasi tinggi dengan peubah target
- Melihat pengaruh setiap fitur numerik terhadap target -> fitur Rainfall_mm memiliki pengaruh tinggi dengan peubah target
- Melihat interaksi antar fitur (Distribusi Yield per Region dan Crop, Interaksi Rainfall dan Temperature dengan warna Yield, Rainfall vs Yield, dipisah berdasarkan Irrigation Used, Frekuensi Kombinasi Region dan Soil_Type)
- Deteksi pencilan -> terdapat pencilan pada peubah target, tetapi tidak dilakukan penanganan untuk membuang data tersebut karena akan kehilangan beberapa informasi penting. 

## Data Preparation
a. Memisahkan fitur dan target = supaya tidak terjadi data leakage yang dapat menyebabkan model overfitting

b. Encoding fitur kategorik = karena model regresi tidak dapat memproses data dalam bentuk kategorik sehingga harus diubah dulu ke numerik

c. Standarisasi fitur numerik menggunakan standarscaler = mengubah fitur numerik agar memiliki rata-rata 0 dan standar deviasi 1

d. Memisahkan data training dan data testing dengan proporsi 80% dan 20% = data training digunakan untuk melatih model dan data testing digunakan untuk mengukur performa model terhadap data yang belum pernah dilihat sebelumnya

## Modeling
1. Linear Regression
Linear Regression adalah metode statistik yang digunakan untuk memodelkan hubungan linear antara variabel input (fitur) dan output (target). Model ini mencoba mencari garis lurus terbaik yang meminimalkan selisih kuadrat antara nilai aktual dan nilai prediksi (disebut metode least squares).

- Library: LinearRegression() dari sklearn.linear_model
- Parameter: Default (tanpa tuning)

- Kelebihan :
    - Sederhana dan mudah diinterpretasikan
    - Cepat untuk dijalankan
- Kekurangan :
    - Asumsi linearitas antara fitur dan target
    - Kurang cocok jika data memiliki hubungan non-linear atau interaksi kompleks.

2. Random Forest Regressor
Random Forest adalah algoritma ensemble learning berbasis pohon keputusan (decision tree). Model ini membangun banyak pohon keputusan secara acak dari subset data dan fitur, lalu menggabungkan hasil prediksi masing-masing pohon (dalam regresi: mengambil rata-rata prediksi). Keunggulannya adalah mampu mengurangi overfitting dan menangani data yang kompleks.

    a. Model ditingkatkan menggunakan Halving Random Search CV, yang mempercepat proses pencarian hyperparameter optimal dengan efisien, khususnya pada data yang berukuran besar.

    Best model Random Forest menggunakan Halving Random Forest didapatkan parameter sebagai berikut:
    n_estimators: 100,
min_samples_split: 5, 
min_samples_leaf: 1,
 max_depth: 20

    b. Mencoba modelling menggunakan feature important random forest dan didapatkan best parameter sebagai berikut:
bootstrap: True, 
ccp_alpha: 0.0, 
criterion: 'squared_error', 
max_depth: 10, 
max_features: 1.0, 
max_leaf_nodes: None, 
max_samples: None, 
min_impurity_decrease: 0.0, 
min_samples_leaf: 3, 
min_samples_split: 5, 
min_weight_fraction_leaf: 0.0, 
monotonic_cst: None, 
n_estimators: 100, 
n_jobs: -1, 
oob_score: False,
random_state: 42, 
verbose: 0, 
warm_start: False

    - Kelebihan:
        - Mampu menangani data non-linear.
        - Tahan terhadap overfitting berkat mekanisme bagging.
        - Dapat menangani missing values dan fitur kategorik (dengan preprocessing).
    - Kekurangan:
        - Interpretabilitas rendah.
        - Waktu pelatihan lebih lama dibanding regresi linier.

3. XGBoost Regressor
XGBoost (Extreme Gradient Boosting) adalah algoritma boosting yang membangun model secara bertahap. Setiap model baru berusaha memperbaiki kesalahan dari model sebelumnya. XGBoost sangat efisien dan sering menghasilkan performa tinggi, terutama pada data tabular. XGBoost menggunakan pohon keputusan sebagai base learner dan optimasi fungsi loss menggunakan teknik gradient descent.

    a. Sama seperti Random Forest, tuning dilakukan dengan Halving Random Search CV untuk meningkatkan efisiensi waktu pelatihan pada data berukuran besar.

    Model didapatkan dengan parameter terbaik sebagai berikut:
subsample: 1.0,
n_estimators: 100,
max_depth: 3,
learning_rate: 0.05

    -  Kelebihan:
        - Sangat kuat untuk data tabular
        - Mendukung regularisasi untuk mengurangi overfitting.
        - Cepat dan efisien dalam proses pelatihan.
    - Kekurangan:
        - Lebih kompleks untuk disetel dan diinterpretasi.
        - Membutuhkan tuning hyperparameter agar performa optimal.

    Pemilihan model terbaik di dasarkan pada pengukuran kebaikan model yaitu RMSE, MAE, dan R Square

## Evaluation
Dalam proyek ini, tujuan utamanya adalah memprediksi hasil panen (yield) dalam satuan ton/hektar, sehingga pendekatan yang digunakan adalah regresi. Oleh karena itu, metrik evaluasi yang dipilih harus mampu mengukur seberapa jauh prediksi model dari nilai aktual. Metrik yang digunakan adalah:

1. **Root Mean Squared Error (RMSE)**
RMSE mengukur akar dari rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual.

    ![alt text](image.png)

2. **Mean Absolute Error (MAE)**
 MAE menghitung rata-rata dari nilai absolut selisih antara prediksi dan aktual.

    ![alt text](image-1.png)

3. **R-squared (R²)**
R² menunjukkan proporsi variabilitas target yang bisa dijelaskan oleh fitur-fitur input.

    ![alt text](image-2.png)

**Hasil Evaluasi Model**

Perbandingan RMSE, MAE, dan R² (Train vs Test) 

Model                        | RMSE Train |  RMSE Test |  MAE Train |   MAE Test |  R² Train |   R² Test
------------------------------------------------------------------------------------------

Linear Regression            |     0.5004 |     0.5008 |     0.3992 |     0.3996 |    0.9130 |    0.9130

Random Forest                |     0.3562 |     0.5088 |     0.2860 |     0.4059 |    0.9559 |    0.9102

**RF (Important Features)      |     0.4973 |     0.5016 |     0.3968 |     0.4002 |    0.9140 |    0.9127**

XGBoost                      |     0.5066 |     0.5074 |     0.4041 |     0.4048 |    0.9108 |    0.9107

Berdasarkan ketiga metrik tersebut:
- Random Forest dengan fitur penting menunjukkan performa terbaik secara konsisten dengan RSME dan MAE yang lebih rendah menandakkan bahwa prediksi dekat dengan nilai aktual dan nilai R² tertinggi (0.9127) menunjukkan bahwa model mampu menjelaskan lebih banyak variasi dari data.

**Persamaan Regresi Logistik**

Koefisien Linear Regression:

Feature  | Coefficient
            
Fertilizer_Used_True   |   1.499406
Rainfall_mm  |   1.298387
Irrigation_Used_True  |   1.199223
Temperature_Celsius   |  0.143947
Soil_Type_Clay   |  0.003190
Soil_Type_Sandy  |   0.002746
Crop_Rice   |  0.001476
Weather_Condition_Rainy   |  0.001454
Days_to_Harvest   |  0.000527
Region_North  |   0.000454
Soil_Type_Loam  |   0.000178
Soil_Type_Peaty  |   0.000071
Weather_Condition_Sunny  |  -0.000249
Region_South  |  -0.000479
Crop_Soybean   | -0.000728
Region_West   | -0.000749
Soil_Type_Silt  |  -0.001662
Crop_Cotton  |  -0.001722
Crop_Wheat  |  -0.002918
Crop_Maize  |  -0.003013

Interpretasi : Berdasarkan koefisien Linear Regression, fitur yang paling berpengaruh positif terhadap hasil panen adalah penggunaan pupuk (+1.5 ton/ha), curah hujan (+1.3 ton/ha), dan penggunaan irigasi (+1.2 ton/ha). Sementara itu, jenis tanaman seperti jagung dan gandum cenderung memiliki kontribusi negatif kecil terhadap prediksi hasil panen.

**Feature Important Random Forest**

Daftar Fitur Penting berdasarkan Feature Importance dari Random Forest:

Feature | Importance

Rainfall_mm  |  0.623394

Fertilizer_Used_True  |  0.204293

Irrigation_Used_True  |  0.130245

Temperature_Celsius  |  0.020315

Days_to_Harvest   | 0.009242

Weather_Condition_Rainy  |  0.000912

Weather_Condition_Sunny  |  0.000911

Region_North  |  0.000867

Region_West  |  0.000865

Region_South  |  0.000863

Crop_Rice  |  0.000820

Crop_Cotton  |  0.000817

Soil_Type_Silt  |  0.000814

Soil_Type_Sandy  |  0.000812

Soil_Type_Peaty  |  0.000810

Soil_Type_Clay  |  0.000808

Soil_Type_Loam  |  0.000807

Crop_Maize |   0.000804

Crop_Soybean   | 0.000801

Crop_Wheat  |  0.000800

Interpretasi : Berdasarkan feature importance dari model Random Forest, fitur Rainfall_mm (62%), Fertilizer_Used (20%), dan Irrigation_Used (13%) adalah prediktor paling penting terhadap hasil panen. Ketiga fitur ini menyumbang lebih dari 95% total kontribusi dalam akurasi model, sedangkan fitur lain seperti suhu, jenis tanaman, dan tanah memiliki pengaruh yang jauh lebih kecil.

**---Ini adalah bagian akhir laporan---**
