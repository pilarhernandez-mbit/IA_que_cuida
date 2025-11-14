# scripts/train_intents.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# === Cargar dataset ===

df = pd.read_csv("./data/intents_train_expanded.csv")
print("Dataset:", df.shape)
print(df["intent"].value_counts())

# === Preparar datos ===
X = df["text"]
y = df["intent"]
le = LabelEncoder()
y_enc = le.fit_transform(y)

vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)
X_vec = vec.fit_transform(X)

# === Entrenar modelo ===
clf = LogisticRegression(max_iter=1000)
clf.fit(X_vec, y_enc)

# === Evaluar ===
Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=0.2, random_state=42)
clf.fit(vec.transform(Xtr), ytr)
preds = clf.predict(vec.transform(Xte))
print(classification_report(yte, preds, target_names=le.classes_))

# === Guardar artefactos ===
os.makedirs("../models", exist_ok=True)
joblib.dump(clf, "../models/intent_clf.joblib")
joblib.dump(le, "../models/label_encoder.joblib")
joblib.dump(vec, "../models/vectorizer.joblib")

print("✅ Modelo entrenado y guardado en carpeta 'models/'")
