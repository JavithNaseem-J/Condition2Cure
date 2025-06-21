import streamlit as st
import pandas as pd
from pathlib import Path
from Condition2Cure.pipeline.predictionpipeline import PredictionPipeline

st.set_page_config(page_title="Patient Condition Classifier", page_icon="💊")
st.title('🧠 Patient Condition Classifier + Drug Recommender 💊')
st.markdown("Predict patient condition based on input description and suggest top-rated drugs.")

pipeline = PredictionPipeline()

@st.cache_data
def load_data():
    df = pd.read_csv("artifacts/data_ingestion/Drugs_Data.csv")
    df.columns = df.columns.str.strip()
    return df

data = load_data()

# UI Input
input_text = st.text_area("📝 Enter patient's description", height=150, placeholder="e.g. I've had a persistent cough and trouble breathing for weeks...")

# Function: Recommend drugs
def recommend_drugs(condition: str, data: pd.DataFrame):
    filtered = data[
        (data['condition'] == condition) &
        (data['rating'] >= 9) &
        (data['usefulCount'] >= 100)
    ]
    top_drugs = (
        filtered.sort_values(['rating', 'usefulCount'], ascending=False)['drugName'].drop_duplicates().head(3).tolist())
    return top_drugs

if st.button("🔍 Predict Condition"):
    if input_text.strip():
        condition = pipeline.predict(input_text)
        st.success(f"🧾 Predicted Condition: **{condition}**")

        drugs = recommend_drugs(condition, data)
        if drugs:
            st.subheader("💊 Top Recommended Drugs:")
            for i, drug in enumerate(drugs, 1):
                st.markdown(f"{i}. **{drug}**")
        else:
            st.warning("⚠️ No top-rated drugs found for this condition.")
    else:
        st.error("Please enter a valid patient description.")
