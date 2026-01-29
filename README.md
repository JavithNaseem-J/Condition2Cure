# 🧠 Condition2Cure

**AI-powered medical condition classifier that predicts diseases from patient symptoms and recommends drugs.**

> 📍 Entry-level ML/Data Science Portfolio Project

---

## 🎯 What This Project Does

1. **Takes** a patient's symptom description (text)
2. **Converts** text to numerical features using BERT embeddings
3. **Predicts** the medical condition using XGBoost
4. **Recommends** top-rated drugs for that condition

---

## 🛠️ Tech Stack

| Component | Technology | Why? |
|-----------|------------|------|
| **Text Embeddings** | Sentence Transformers (BERT) | Captures semantic meaning better than TF-IDF |
| **ML Model** | XGBoost | Fast, accurate, works well with embeddings |
| **Hyperparameter Tuning** | Optuna | Finds optimal parameters automatically |
| **Web App** | Streamlit | Simple, clean Python web framework |
| **Experiment Tracking** | MLflow | Track model versions and metrics |
| **Data Versioning** | DVC | Version control for datasets |

---

## 📁 Project Structure

```
Condition2Cure/
├── app.py                 # Streamlit web application
├── main.py                # Training pipeline runner
├── requirements.txt       # Dependencies
│
├── src/Condition2Cure/
│   ├── components/        # Core ML components
│   │   ├── data_cleaning.py
│   │   ├── data_transformation.py   # BERT embeddings
│   │   ├── model_training.py        # XGBoost + Optuna
│   │   └── model_evaluation.py
│   │
│   ├── pipeline/
│   │   ├── feature_pipeline.py      # Data processing
│   │   ├── model_pipeline.py        # Training
│   │   └── predictionpipeline.py    # Inference
│   │
│   └── utils/
│       └── nlp_utils.py             # Text processing
│
├── artifacts/             # Saved models and data
├── config/                # Configuration files
└── mlruns/                # MLflow experiments
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training Pipeline

```bash
python main.py
```

### 3. Start Web App

```bash
streamlit run app.py
```

---

## 💡 Key Concepts Explained

### Why Sentence Transformers instead of TF-IDF?

**TF-IDF** counts word frequency. "headache" and "head pain" are completely different.

**BERT** understands meaning. "headache" and "head pain" have similar embeddings because they mean the same thing.

```python
# Old way (TF-IDF)
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(texts)  # Sparse, word-based

# New way (BERT)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
features = model.encode(texts)  # Dense, semantic
```

### Why XGBoost with BERT embeddings?

- BERT creates 384-dim vectors capturing meaning
- XGBoost classifies these vectors efficiently
- **No GPU needed** for inference (unlike fine-tuning BERT)
- Fast predictions (~5ms vs ~100ms for BERT classifier)

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| F1 Score (weighted) | ~0.85 |
| Accuracy | ~0.83 |
| Inference Time | <10ms |

---

## 🔧 How to Retrain

```bash
# Run full pipeline
python main.py

# Or run individual stages
python main.py --stage feature_pipeline
python main.py --stage model_pipeline
```

---

## 📝 Interview Talking Points

1. **Why BERT over TF-IDF?** 
   - Semantic understanding vs word counting
   - "headache" ≈ "head pain" in BERT space

2. **Why not fine-tune BERT directly?**
   - Requires GPU, slower inference
   - XGBoost on embeddings = best of both worlds

3. **Why Optuna?**
   - Smarter than grid search (Bayesian optimization)
   - Automatically finds best hyperparameters

4. **Why MLflow?**
   - Tracks experiments, metrics, model versions
   - Essential for production ML

---

## ⚠️ Disclaimer

This is an **educational project**. Not for real medical diagnosis.

---

## 📫 Contact

**[Your Name]**  
📧 your.email@example.com  
💼 [LinkedIn](https://linkedin.com/in/yourprofile)  
🐙 [GitHub](https://github.com/yourusername)