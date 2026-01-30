import streamlit as st
import pandas as pd
from Condition2Cure.pipeline.predictionpipeline import PredictionPipeline

# Page setup
st.set_page_config(
    page_title="Condition2Cure",
    page_icon="💊",
    layout="centered"
)

st.title("🧠 Condition2Cure")
st.markdown("**AI-powered medical condition classifier + drug recommender**")
st.markdown("---")


# Load model (cached so it only loads once)
@st.cache_resource
def load_model():
    return PredictionPipeline()


@st.cache_data
def load_drug_data():
    df = pd.read_csv("artifacts/data_ingestion/Drugs_Data.csv")
    df.columns = df.columns.str.strip()
    return df


# Initialize
try:
    pipeline = load_model()
    drug_data = load_drug_data()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


def get_drug_recommendations(condition, data, top_k=3):
    """Get top-rated drugs for a condition."""
    filtered = data[
        (data['condition'].str.lower() == condition.lower()) &
        (data['rating'] >= 8) &
        (data['usefulCount'] >= 50)
    ]
    
    if filtered.empty:
        filtered = data[data['condition'].str.lower() == condition.lower()]
    
    top_drugs = (
        filtered
        .sort_values(['rating', 'usefulCount'], ascending=False)
        .drop_duplicates(subset=['drugName'])
        .head(top_k)
    )
    
    return top_drugs[['drugName', 'rating', 'usefulCount']].to_dict('records')


# User input
st.subheader("📝 Enter Patient Symptoms")
user_input = st.text_area(
    "Describe the symptoms:",
    height=150,
    placeholder="Example: I've been having severe headaches and feeling nauseous for the past week..."
)

# Prediction button
if st.button("🔍 Predict Condition", type="primary"):
    if user_input.strip() and len(user_input) >= 10:
        
        with st.spinner("Analyzing symptoms..."):
            # Get prediction
            condition, confidence = pipeline.predict(user_input)
        
        # Show results
        st.markdown("---")
        st.subheader("📋 Results")
        
        # Prediction box
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Condition", condition)
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        
        # Confidence indicator
        if confidence >= 0.7:
            st.success("✅ High confidence prediction")
        elif confidence >= 0.4:
            st.warning("⚠️ Moderate confidence - consider consulting a doctor")
        else:
            st.error("❌ Low confidence - please consult a healthcare provider")
        
        # Drug recommendations
        st.markdown("---")
        st.subheader("💊 Recommended Drugs")
        
        drugs = get_drug_recommendations(condition, drug_data)
        
        if drugs:
            for i, drug in enumerate(drugs, 1):
                st.write(f"**{i}. {drug['drugName']}** - Rating: {drug['rating']}/10 ({drug['usefulCount']} reviews)")
        else:
            st.info("No drug recommendations found for this condition.")
        
        # Show alternatives if confidence is low
        if confidence < 0.6:
            st.markdown("---")
            st.subheader("🔄 Other Possible Conditions")
            alternatives = pipeline.predict_top_k(user_input, k=5)
            for cond, conf in alternatives:
                st.write(f"- {cond}: {conf:.1%}")
    else:
        st.warning("Please enter a longer description (at least 10 characters)")

# Footer
st.markdown("---")
st.caption("⚠️ This is for educational purposes only. Always consult a healthcare professional.")
