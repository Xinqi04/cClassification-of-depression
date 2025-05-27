# **Laporan Proyek Machine Learning - Riza Anwar Fadil**

---

## Domain Proyek: Kesehatan Mental Siswa

### Latar Belakang

Kesehatan mental merupakan aspek penting dalam kehidupan seorang Siswa, terutama di tengah tekanan akademik, sosial, dan pribadi. Menurut World Health Organization (WHO), lebih dari 264 juta orang di seluruh dunia mengalami depresi, termasuk Siswa usia produktif. Sebuah studi oleh American College Health Association (2020) menyatakan bahwa lebih dari 40% Siswa mengalami gejala depresi selama masa studi mereka.

Siswa sering kali mengalami tekanan dari tugas, jadwal yang padat, dan tekanan sosial, sehingga membuat mereka rentan terhadap gangguan kesehatan mental. Oleh karena itu, identifikasi dini terhadap gejala depresi menjadi krusial agar dapat diberikan penanganan atau pendampingan yang tepat.

Masalah ini penting diselesaikan karena gangguan kesehatan mental yang tidak ditangani dapat menyebabkan penurunan prestasi akademik, ketidakhadiran kuliah, hingga risiko bunuh diri.

Referensi:

- (https://www.researchgate.net/profile/Nurul-Hidayah-45/publication/343991731_KEPEKAAN_HUMOR_DENGAN_DEPRESI_PADA_REMAJA_DITINJAU_DARI_JENIS_KELAMIN/links/6018097545851517ef2f2867/KEPEKAAN-HUMOR-DENGAN-DEPRESI-PADA-REMAJA-DITINJAU-DARI-JENIS-KELAMIN.pdf)
- (https://www.who.int/news-room/fact-sheets/detail/depression)
- (https://journalthamrin.com/index.php/jikmht/article/view/422)

---

## Business Understanding

### Problem Statements:

1. Bagaimana cara mengklasifikasikan Siswa yang berpotensi mengalami depresi berdasarkan data survei?
2. Faktor-faktor apa saja yang paling berpengaruh terhadap tingkat depresi Siswa?

### Goals:

1. Mengembangkan model klasifikasi untuk mendeteksi Siswa yang berpotensi mengalami depresi.
2. Mengidentifikasi fitur atau atribut paling signifikan dalam mendeteksi potensi depresi.

### Solution Statements:

- Membangun model klasifikasi menggunakan **Logistic Regression**, **Random Forest**, dan **Support Vector Machine (SVM)**.
- Melakukan **feature selection** dan **hyperparameter tuning** untuk meningkatkan performa model.
- Mengukur performa model dengan metrik: **accuracy, precision, recall, dan F1-score**.

---

## Data Understanding

Dataset yang digunakan diunduh dari Kaggle dengan nama **Student Depression Dataset**.
📎 [Link Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)

### Penjelasan Variabel:

Beberapa fitur dalam dataset:

- id: Identifikasi unik responden.
- Gender: Jenis kelamin.
- Age: Umur responden.
- City: Kota tempat tinggal.
- Profession: Pekerjaan atau status (misalnya, student).
- Academic Pressure: Tingkat tekanan akademis.
- Work Pressure: Tingkat tekanan dari pekerjaan.
- CGPA: Indeks Prestasi Kumulatif (IPK).
- Study Satisfaction: Tingkat kepuasan belajar.
- Job Satisfaction: Tingkat kepuasan kerja.
- Sleep Duration: Durasi tidur responden.
- Dietary Habits: Kebiasaan makan.
- Degree: Gelar pendidikan.
- Have you ever had suicidal thoughts?: Indikator pikiran bunuh diri (Yes/No).
- Work/Study Hours: Jumlah jam kerja/studi per minggu.
- Financial Stress: Tingkat stres finansial.
- Family History of Mental Illness: Riwayat penyakit mental dalam keluarga.
- Depression: Status depresi (target: Yes/No).

Dataset memiliki **27901 baris** dan **18 kolom**.

_Melakukan visualisasi dan EDA_

- Distribusi Label Target (Depression)
- Perbandingan Depresi Berdasarkan Gender
- Hubungan Akademik dan Depresi
- Pola Tidur dan Depresi
- Tekanan Finansial dan Depresi
- Riwayat penyakit mental dalam keluarga dan Depresi
- Dietary Habits dan Depresi
- Degree dan Depresi
- Pemikiran Bunuh Diri dan Depresi
- Korelasi Antar Fitur Numerik

---

## Data Preparation

### Langkah-langkah:

1. **Mising Values & Duplicates**
   Ditemukan 3 nilai kosong (missing values) pada kolom Financial Stress. Nilai-nilai tersebut diimputasi menggunakan nilai median, karena distribusinya bersifat skewed dan median lebih tahan terhadap outlier dibandingkan mean. Tidak ditemukan data duplikat dalam dataset.
2. **Feature selection**
   Pemilihan fitur dilakukan berdasarkan analisis bisnis, di mana fokus analisis ditujukan pada mahasiswa (student). Oleh karena itu, data yang digunakan hanya berasal dari individu dengan profesi student.
   Beberapa kolom dihapus karena dianggap tidak memberikan kontribusi signifikan terhadap prediksi variabel target (Depression), yaitu:

- id: hanya sebagai identitas unik, tidak relevan untuk pemodelan.
- City: terlalu spesifik dan berpotensi menambah dimensi tanpa makna yang substansial.
- Degree: tidak menunjukkan korelasi kuat terhadap Depression dan berpotensi menimbulkan multikolinearitas jika dilakukan encoding.

3. **Encoding**
   Dilakukan encoding terhadap kolom-kolom kategorikal sebagai berikut

- Ordinal Encoding pada kolom Sleep Duration dan Dietary Habits karena memiliki urutan yang logis.
- Binary Encoding pada kolom Gender, Have you ever had suicidal thoughts?, dan Family History of Mental Illness karena hanya memiliki dua kategori.

4. **Train-test split**
   Data dipisahkan antara fitur dan target, dengan Depression sebagai variabel target. Kemudian dilakukan pembagian data menjadi data latih dan data uji dengan rasio 80:20, untuk memastikan model dapat dievaluasi secara adil pada data yang belum pernah dilihat sebelumnya.

---

## **Modeling**

Tiga algoritma machine learning digunakan sebagai pendekatan, yaitu Logistic Regression, Random Forest Classifier, dan Support Vector Machine (SVM). Pemilihan model ini didasarkan pada karakteristik data dan keunggulan masing-masing algoritma.

### 1. Logistic Regression

- Model linier yang digunakan sebagai baseline karena kesederhanaannya dan kemampuannya untuk memberikan interpretasi yang jelas terhadap koefisien masing-masing fitur.
- Parameter:
  - `solver='liblinear'`: Solver yang cocok untuk dataset kecil hingga menengah dan binary classification.
  - `random_state=42`: Untuk menjaga reproducibility.

### 2. Random Forest Classifier

- Model ensemble berbasis pohon keputusan yang membentuk banyak decision tree dan menggabungkan hasilnya (bagging) untuk meningkatkan akurasi dan mengurangi overfitting.
- Parameter:
  - `n_estimators=100`: Jumlah pohon dalam hutan (default umum dan cukup stabil).
  - `random_state=42`: Untuk menjaga konsistensi hasil.

### 3. Support Vector Machine (SVM)

- Algoritma klasifikasi yang bekerja dengan memisahkan kelas data menggunakan hyperplane optimal. SVM efektif dalam ruang berdimensi tinggi.
- Parameter:
  - `kernel='rbf'`: Kernel Gaussian (Radial Basis Function), cocok untuk data non-linear.
  - `probability=True`: Mengaktifkan prediksi probabilitas (digunakan dalam evaluasi ROC AUC).
  - `random_state=42`: Konsistensi hasil antar run.

```python
models = {
    'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}
```

Setiap model dilatih menggunakan data pelatihan (`X_train`, `y_train`) dan kemudian diuji performanya pada data pengujian (`X_test`, `y_test`). Evaluasi dilakukan menggunakan berbagai metrik seperti accuracy, precision, recall, f1-score, dan ROC AUC untuk memastikan kualitas model secara menyeluruh.

---

## **Evaluation**

### **Metrik Evaluasi:**

- Accuracy = jumlah prediksi benar / total data
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-score = 2 × (Precision × Recall) / (Precision + Recall)
- ROC AUC = Area under the ROC Curve, mengukur kemampuan model membedakan kelas secara umum.

### Hasil Evaluasi Model:

| Model               | Accuracy | Precision | Recall | F1-score | ROC AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 84.63%   | 85.00%    | 89.19% | 87.04%   | 92.19%  |
| Random Forest       | 84.12%   | 85.10%    | 87.98% | 86.52%   | 91.48%  |
| SVM                 | 84.68%   | 84.89%    | 89.46% | 87.12%   | 92.20%  |

### Kesimpulan:

- SVM menunjukkan kinerja terbaik secara keseluruhan dengan skor tertinggi pada Accuracy, Recall, F1-score, dan ROC AUC.
- Logistic Regression berada sangat dekat dengan performa SVM, terutama dalam hal ROC AUC.
- Random Forest juga memiliki hasil yang sangat baik dan unggul sedikit dalam hal Precision, namun sedikit lebih rendah dalam F1-score dan ROC AUC.

### Model Terbaik:

- Berdasarkan evaluasi di atas, SVM (Support Vector Machine) dipilih sebagai model terbaik karena memberikan performa paling konsisten dan unggul di sebagian besar metrik utama.

## Fine-Tuning dan Feature Selection: SVM

Melakukan penyetelan hyperparameter (fine-tuning) dan seleksi fitur (feature selection) pada model Support Vector Machine (SVM) untuk meningkatkan performa klasifikasi.

### Fine Tuning

Dilakukan dengan menggunakan GridSearchCV untuk mencari kombinasi terbaik dari hyperparameter SVM, yaitu:

- C: [0.1, 1, 10] – parameter regularisasi
- gamma: ['scale', 'auto', 0.01, 0.001] – koefisien kernel RBF
- kernel: ['rbf'] – kernel non-linear RBF

### Feature Selection

Menggunakan SelectKBest dengan f_classif (ANOVA F-value) untuk memilih fitur yang paling relevan terhadap target. Dicoba beberapa nilai k yaitu: [5, 10, 'all'].

### Hasil

- Best Parameters:

  ```python
  {
      'feature_selection__k': 10,
      'svm__C': 1,
      'svm__gamma': 'scale',
      'svm__kernel': 'rbf'
  }
  ```

- Fitur Terpilih (Top 10):
  1. Age
  2. Academic Pressure
  3. CGPA
  4. Study Satisfaction
  5. Sleep Duration
  6. Dietary Habits
  7. Have you ever had suicidal thoughts?
  8. Work/Study Hours
  9. Financial Stress
  10. Family History of Mental Illness

---

### Evaluasi Performa Model

| Metric        | Nilai |
| ------------- | ----- |
| Accuracy      | 0.85  |
| Precision (0) | 0.84  |
| Precision (1) | 0.85  |
| Recall (0)    | 0.78  |
| Recall (1)    | 0.89  |
| F1-score (0)  | 0.81  |
| F1-score (1)  | 0.87  |
| Macro F1      | 0.84  |
| Weighted F1   | 0.85  |

---

### Kesimpulan

Tidak terjadi perubahan signifikan antara hasil fine tuning dan sebelum nya.

---

## Penutup

Proyek ini mengembangkan model klasifikasi untuk mendeteksi potensi depresi pada siswa berdasarkan data survei yang mencakup faktor akademik, sosial, dan kebiasaan hidup. Tiga model telah diuji—Logistic Regression, Random Forest, dan Support Vector Machine (SVM)—dengan SVM terpilih sebagai model terbaik berdasarkan metrik evaluasi seperti accuracy, recall, F1-score, dan ROC AUC.

Dari hasil evaluasi, model menunjukkan performa yang sangat baik dengan nilai F1-score mencapai 0.87 pada kelas target, yang mengindikasikan kemampuan deteksi yang kuat terhadap siswa yang berisiko depresi. Proses feature selection juga berhasil mengidentifikasi 10 atribut paling signifikan, seperti Academic Pressure, Study Satisfaction, Financial Stress, dan Family History of Mental Illness, yang memberikan wawasan penting mengenai faktor-faktor yang berkontribusi terhadap kondisi mental siswa.

Model ini menunjukkan potensi besar dalam membantu lembaga pendidikan atau pihak konselor untuk melakukan identifikasi dini terhadap siswa yang berisiko mengalami gangguan kesehatan mental, sehingga dapat dilakukan intervensi lebih lanjut secara preventif.

---
