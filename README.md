
# AI Spam Message Detector

Aplikasi Machine Learning untuk mengklasifikasikan pesan SMS menjadi Spam dan Not Spam.

## Tentang Proyek

Proyek ini dibuat menggunakan Python dan Streamlit untuk membantu mengidentifikasi pesan yang berpotensi spam berdasarkan pola teks.

## Teknologi yang Digunakan

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes

## Dataset

SMS Spam Collection Dataset.

## Model Evaluation

- Accuracy: 97.04%
- Precision (Spam): 1.00
- Recall (Spam): 0.78
- F1-Score (Spam): 0.88

## Struktur Proyek

```text
AI-Spam-Message-Detector/
│
├── app.py
├── train_model.py
├── spam_model.pkl
├── requirements.txt
├── README.md
│
└── data/
    └── SMSSpamCollection.tsv
```

## Cara Menjalankan

1. Install library:

```bash
pip install -r requirements.txt
```

2. Jalankan aplikasi:

```bash
streamlit run app.py
```

## Catatan

Model dilatih menggunakan pesan SMS berbahasa Inggris.
Hasil prediksi merupakan estimasi model dan bukan jaminan keamanan pesan.

## Author

Machine Learning Portfolio Project
