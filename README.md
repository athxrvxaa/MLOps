# 🚀 MLOps Pipeline with ZenML

> An end-to-end **Machine Learning pipeline** built using **ZenML** to understand core **MLOps concepts** such as modular pipelines, reproducibility, and experiment tracking.

---

## 📌 Overview

This repository contains my **first MLOps project using ZenML**.  
The project demonstrates how to build a **production-style ML pipeline** with clearly defined steps for:

- 📥 Data ingestion  
- 🧹 Data cleaning  
- 🤖 Model training  
- 📊 Model evaluation  

All components are orchestrated using **ZenML pipelines**.

---

## 📁 Project Structure

```text
.
├── .zen/                   # ZenML metadata & configs
├── data/                   # Datasets (raw / processed)
├── pipelines/              # Pipeline definitions
│   └── training_pipeline.py
├── src/                    # Core ML logic
│   ├── data_cleaning.py
│   ├── model_dev.py
│   └── eval.py
├── steps/                  # ZenML pipeline steps
│   ├── ingest_data.py
│   ├── clean_data.py
│   ├── model_train.py
│   ├── evaluation.py
│   └── config.py
├── saved_model/            # Trained model artifacts
├── run_pipeline.py         # Pipeline entry point
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

```
## 🧠 Features

- ✔ Modular pipeline design  
- ✔ Reproducible ML workflow  
- ✔ ZenML step-based architecture  
- ✔ Clean project structure  
- ✔ Easy experimentation  

---

## 🛠️ Tech Stack

- **Python**
- **ZenML**
- **Scikit-learn**
- **Pandas**
- **NumPy**

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd mlops-mine


⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone <your-repo-url>
cd mlops-mine

2️⃣ Create & Activate Virtual Environment (Recommended)
python -m venv mlops_env
source mlops_env/bin/activate   # macOS / Linux
mlops_env\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Initialize ZenML
zenml init

▶️ Running the Pipeline
python run_pipeline.py

```

### This command will:

- 📥 Ingest data
- 🧹 Clean & preprocess data
- 🤖 Train a model
- 📊 Evaluate performance

---

## 📊 Outputs

- 🧠 Trained models stored in `saved_model/`
- 📈 Metrics logged during evaluation
- 🧾 Pipeline runs tracked using **ZenML**

---

## 🎯 Learning Goals

- Understand **MLOps fundamentals**
- Learn **ZenML pipelines & steps**
- Practice **clean ML project structuring**
- Build **reproducible ML systems**

---

## 🚧 Future Enhancements

- 🔍 Experiment tracking (MLflow)
- 🚀 Model deployment
- 🔄 CI/CD integration
- 📑 Data validation
- 📈 Advanced evaluation metrics
