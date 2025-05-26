import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# === STEP 1: Load All Datasets ===
print("📥 Loading datasets...")

df_main = pd.read_csv("data/dataset.csv")
df_severity = pd.read_csv("data/Symptom-severity.csv")
df_desc = pd.read_csv("data/symptom_Description.csv")
df_prec = pd.read_csv("data/symptom_precaution.csv")

# === STEP 2: Normalize All Column Names ===
for df in [df_main, df_severity, df_desc, df_prec]:
    df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]

# === STEP 3: Normalize symptom names in severity file ===
if "symptom" in df_severity.columns:
    df_severity["symptom"] = df_severity["symptom"].str.strip().str.lower().str.replace(" ", "_")

# === STEP 4: Combine Symptom Columns into One Free-Text Column ===
symptom_cols = [col for col in df_main.columns if col.startswith("symptom")]

def merge_symptoms(row):
    symptoms = [str(s).strip().lower().replace(" ", "_") for s in row[symptom_cols] if pd.notna(s)]
    return ", ".join(symptoms)

df_main["symptoms"] = df_main.apply(merge_symptoms, axis=1)

# Final training DataFrame
df_train = df_main[["symptoms", "disease"]].dropna()

# === STEP 5: Split Data ===
X = df_train["symptoms"]
y = df_train["disease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 6: Build and Train Model ===
print("🧠 Training TF-IDF + Logistic Regression...")
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])
model.fit(X_train, y_train)

# === STEP 7: Evaluate ===
print("\n📊 Model Evaluation:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# === STEP 8: Save Model ===
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/symptom_model.pkl")
print("✅ Model saved to model/symptom_model.pkl")
