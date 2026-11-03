
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Menentukan lokasi dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection.tsv"

# 2. Membaca dataset
data = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# 3. Membersihkan data
data = data.dropna()
data["label"] = data["label"].str.lower().str.strip()

print("Jumlah data:", len(data))
print("\nDistribusi label:")
print(data["label"].value_counts())

# 4. Memisahkan fitur dan target
X = data["message"].astype(str)
y = data["label"]

# 5. Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Membuat model Machine Learning
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

# 7. Melatih model AI
print("\nSedang melatih model AI...")
model.fit(X_train, y_train)

# 8. Menguji model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===== HASIL EVALUASI MODEL =====")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=["ham", "spam"],
    target_names=["Not Spam", "Spam"],
    zero_division=0
))

# 9. Menyimpan model AI
MODEL_PATH = BASE_DIR / "spam_model.pkl"
joblib.dump(model, MODEL_PATH)

print("\nModel berhasil disimpan!")
print("Lokasi:", MODEL_PATH)
