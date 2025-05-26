import streamlit as st
import joblib
import pandas as pd

# === Load Trained Model ===
model = joblib.load("model/symptom_model.pkl")

# === Load All Datasets ===
desc_df = pd.read_csv("data/symptom_Description.csv")
precaution_df = pd.read_csv("data/symptom_precaution.csv")
severity_df = pd.read_csv("data/Symptom-severity.csv")

# === Normalize Columns ===
for df in [desc_df, precaution_df, severity_df]:
    df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]

severity_df["symptom"] = severity_df["symptom"].str.strip().str.lower().str.replace(" ", "_")

# === Streamlit UI ===
st.set_page_config(page_title="AI Symptom Checker", layout="centered")
st.title("🩺 AI Symptom Checker")
st.markdown("Enter your symptoms (comma-separated). Example: *fever, sore throat, dry cough*")

user_input = st.text_area("Symptoms", height=150, placeholder="Type symptoms here...")

if st.button("Predict Disease"):
    if not user_input.strip():
        st.warning("⚠️ Please enter symptoms.")
    else:
        # === Clean user input ===
        normalized_input = user_input.lower().strip().replace(" ", "_")
        input_symptoms = [s.strip() for s in normalized_input.split(",") if s.strip()]

        # === Model Prediction ===
        probas = model.predict_proba([normalized_input])[0]
        classes = model.classes_
        proba_dict = dict(zip(classes, probas))
        top_3 = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:3]

        prediction, confidence = top_3[0]

        # === Output Prediction ===
        st.subheader(f"🧠 Predicted Disease: **{prediction}**")
        st.write(f"🔬 Confidence Score: **{confidence:.2f}**")

        if confidence < 0.6:
            st.error("⚠️ Low confidence — please consult a doctor.")
        else:
            st.success("✅ This is a confident prediction.")

        # === Top 3 Predictions ===
        st.markdown("### 🔝 Top 3 Possible Diseases:")
        for i, (disease, score) in enumerate(top_3, 1):
            st.markdown(f"{i}. **{disease}** — *{score:.2f}*")

        # === Disease Description ===
        desc_row = desc_df[desc_df['disease'].str.lower() == prediction.lower()]
        if not desc_row.empty:
            st.markdown("📋 **About the Disease:**")
            st.info(desc_row.iloc[0]['description'])

        # === Symptom Severity Info ===
        st.markdown("🌡️ **Severity of Your Reported Symptoms:**")
        for sym in input_symptoms:
            match = severity_df[severity_df['symptom'] == sym]
            if not match.empty:
                level = match.iloc[0]['weight']
                emoji = "🔴" if level >= 4 else "🟠" if level >= 2 else "🟢"
                st.write(f"{emoji} {sym.replace('_', ' ')}: Severity {int(level)}/5")
            else:
                st.write(f"- {sym.replace('_', ' ')}: No severity data found.")

        # === Precautions ===
        prec_row = precaution_df[precaution_df['disease'].str.lower() == prediction.lower()]
        if not prec_row.empty:
            st.markdown("🛡️ **Recommended Precautions:**")
            for i in range(1, 5):
                key = f'precaution_{i}'
                if key in prec_row.columns and prec_row.iloc[0][key]:
                    st.write(f"- {prec_row.iloc[0][key]}")
