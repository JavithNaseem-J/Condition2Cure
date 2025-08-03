# 🧠 Condition2Cure

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
```

---

## ⚙️ Installation

### **Clone the Repository**

```bash
git clone https://github.com/yourusername/Condition2Cure.git
cd Condition2Cure
```

### **Install Dependencies**

```bash
pip install -r requirements.txt
```

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

```bash
python main.py
```

### **Run Specific Stage**

```bash
python main.py --stage feature_pipeline
python main.py --stage model_pipeline
```

### **Launch Streamlit App**

```bash
streamlit run app.py
```

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


