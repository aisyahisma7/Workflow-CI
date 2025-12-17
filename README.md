# Diabetes Prediction – CI with MLflow

Repository ini berisi contoh penerapan workflow CI sederhana untuk proses
training model machine learning menggunakan MLflow dan GitHub Actions.

Project ini dibuat sebagai bagian dari proyek akhir **Membangun Sistem Machine Learning**,
dengan fokus pada otomasi training model setiap kali terjadi perubahan kode.

## Gambaran Umum
Model yang digunakan bertujuan untuk memprediksi kemungkinan diabetes berdasarkan data kesehatan. 
Proses training dijalankan secara otomatis menggunakan MLflow Project melalui pipeline CI.

## Alur Kerja Singkat
1. Kode di-*push* ke repository
2. GitHub Actions menjalankan workflow CI
3. MLflow menjalankan proses training model
4. Hasil training tercatat pada MLflow

## Dataset
Dataset dibagi menjadi dua bagian:
- **Data training** (`diabetes_train.csv`)
- **Data testing** (`diabetes_test.csv`)
Kolom target yang digunakan adalah **Outcome**, di mana nilai `1` menunjukkan diabetes dan `0` menunjukkan tidak diabetes.

## Tools yang Digunakan
- MLflow
- GitHub Actions
- Scikit-learn
- Python
- Docker
