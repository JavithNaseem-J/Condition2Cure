# 🧠 Condition2Cure

<<<<<<< HEAD
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
=======
## Medical Condition Prediction & Drug Recommendation

## 📌 Project Overview

Condition2Cure is an **end-to-end medical NLP solution** designed to predict patient medical conditions from textual descriptions and recommend the most effective drugs. It leverages advanced **Natural Language Processing (NLP)** techniques, **machine learning pipelines**, and **model management tools** to create a production-ready healthcare system.

Our goal is to make accurate condition prediction and drug recommendations accessible for healthcare applications, telemedicine platforms, and research purposes.

---

## ✨ Key Features & Innovations

* **Advanced NLP Processing**: Cleans, processes, and transforms medical reviews and patient descriptions.
* **Condition Prediction Model**: Trained using **XGBoost** with **Optuna hyperparameter tuning**.
* **Drug Recommendation Engine**: Suggests top-rated drugs based on real patient feedback.
* **Data Version Control**: Managed using **DVC** for reproducibility.
* **Experiment Tracking**: **MLflow** integration for metrics, parameters, and artifact logging.
* **Containerized Deployment**: Dockerized for scalable and portable deployment.
* **Interactive UI**: Streamlit app for real-time prediction and recommendation.

---

## 🏗 Technical Architecture

```
Patient Input (Text) → NLP Cleaning → Feature Engineering (TF-IDF + SVD) → XGBoost Model → Predicted Condition → Drug Recommendation
```

### **Pipeline Components**

1. **Data Ingestion**: Downloads and extracts medical review datasets【15†source】.
2. **Data Validation**: Ensures schema correctness【16†source】.
3. **Data Cleaning**: Removes noise and standardizes text【17†source】.
4. **Data Transformation**: TF-IDF vectorization + SVD dimensionality reduction + label encoding【18†source】.
5. **Model Training**: XGBoost with Optuna hyperparameter tuning【20†source】.
6. **Model Evaluation**: Generates metrics and confusion matrix, logs to MLflow【19†source】.
7. **Model Registry**: Promotes best models to production【21†source】.

---

## 📦 Prerequisites

```
- Python 3.8+
- Docker (optional, for containerized deployment)
>>>>>>> b340288e4d8e63370b1b9018c79b409781608950
```

---

<<<<<<< HEAD
## 🚀 Quick Start

### 1. Install Dependencies
=======
## ⚙️ Installation

### **Clone the Repository**

```bash
git clone https://github.com/yourusername/Condition2Cure.git
cd Condition2Cure
```

### **Install Dependencies**
>>>>>>> b340288e4d8e63370b1b9018c79b409781608950

```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
### 2. Run Training Pipeline
=======
### **Docker Setup (Optional)**

```bash
docker build -t condition2cure .
docker run -p 8501:8501 condition2cure
```

---

## 🔧 Configuration

All configuration parameters are stored in `config/config.yaml` and `config/params.yaml`. This includes:

* Data source IDs
* Model parameters
* File paths for artifacts
* Evaluation settings

---

## 🚀 Usage

### **Run Entire Pipeline**
>>>>>>> b340288e4d8e63370b1b9018c79b409781608950

```bash
python main.py
```

<<<<<<< HEAD
### 3. Start Web App
=======
### **Run Specific Stage**

```bash
python main.py --stage feature_pipeline
python main.py --stage model_pipeline
```

### **Launch Streamlit App**
>>>>>>> b340288e4d8e63370b1b9018c79b409781608950

```bash
streamlit run app.py
```

<<<<<<< HEAD
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
=======
Enter a patient description, and the system will:

1. Predict the medical condition.
2. Recommend top-rated drugs for that condition.

---

## 📊 Model Training & Evaluation

* **Algorithm**: XGBoost
* **Tuning**: Optuna
* **Metrics**: Accuracy, Precision, Recall, F1-score
* **Tracking**: MLflow
* **Versioning**: DVC for dataset and pipeline outputs

---

<img width="1919" height="871" alt="image" src="https://github.com/user-attachments/assets/7afb7c13-4647-4c46-b8b9-82051060ab62" />


<img width="1904" height="864" alt="image" src="https://github.com/user-attachments/assets/1732feca-3778-45d3-a22b-ad2020b6c9e3" />




## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


>>>>>>> b340288e4d8e63370b1b9018c79b409781608950
