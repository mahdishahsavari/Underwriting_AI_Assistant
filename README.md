```markdown
# 🔥 AI Underwriting Assistant for Fire Insurance

An end-to-end **AI-powered underwriting and customer intelligence system** designed for the **Fire Insurance** domain.  
This project integrates **risk assessment**, **churn prediction**, and **large language models** to support underwriters and business teams with **data-driven decisions and actionable recommendations**.

---

## 🚀 Project Overview

This project implements a **three-phase intelligent underwriting pipeline** tailored for insurance companies:

1. Risk Assessment & Scoring  
2. Customer Churn Prediction & Analysis  
3. LLM-based Decision Support & Recommendations  

The system follows a real-world underwriting workflow where an insurance case is analyzed from **risk**, **retention**, and **business strategy** perspectives, and finally summarized by a **fine-tuned Large Language Model**.

---

## 🧠 End-to-End Workflow

```

Insurance Case
↓
Phase 1: Risk Scoring & Risk Level Classification
↓
Phase 2: Churn Probability & Retention Insights
↓
Phase 3: LLM-Based Analysis & Recommendations
↓
Final Underwriting Intelligence (JSON + Natural Language)

```

---

## 🧩 Phase 1: Risk Scoring & Risk Level Classification

The first phase predicts the **Risk Score** and **Risk Level** of an insurance case using **multi-class classification models**.

### Models Used
- Gradient Boosting Algorithms  
  - XGBoost  
  - LightGBM  
  - CatBoost  

- Transformer-based Tabular Models  
  - TabNet  
  - TabTransformer  
  - TabM  

### Outputs
- Risk Score  
- Risk Level (multi-class categories)

This phase serves as the **core underwriting risk engine**.

---

## 📉 Phase 2: Customer Churn Prediction & Retention Analysis

The second phase focuses on **customer retention and churn intelligence**.

### Models Used
- LightGBM  
- CatBoost  

### Tasks
- Predict **probability of churn (p_churn)**  
- Perform **churn segmentation**  
- Extract **top churn drivers / reasons**  
- Support retention and underwriting strategies  

---

## 🤖 Phase 3: LLM-Based Decision Support & Recommendations

Outputs from Phase 1 and Phase 2 are aggregated into a structured **JSON** and passed to a **fine-tuned Large Language Model**.

### Fine-Tuned Models
- GPT-OSS (20B parameters) — *best performance*  
- Gemma 3 (4B parameters)  

### LLM Responsibilities
- Holistic analysis of insurance cases  
- Generation of underwriting insights  
- Delivery of **4-step actionable recommendations** for:
  - Risk handling  
  - Policy conditions and pricing  
  - Customer retention  
  - Churn prevention  

The LLM acts as an **AI underwriting assistant**, bridging predictive analytics and business decision-making.

---

## 📦 Tech Stack

- **Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib  
- **Machine Learning:**  
  - XGBoost  
  - LightGBM  
  - CatBoost  
  - TabNet  
  - TabTransformer  
  - TabM  
- **Large Language Models:**  
  - GPT-OSS (20B)  
  - Gemma 3 (4B)  
- **LLM Fine-Tuning & Inference:**  
  - Unsloth  
  - vLLM  

---

## 🔐 Dataset Notice

The dataset used in this project is **confidential and proprietary** and cannot be shared publicly.

All **data cleaning**, **feature engineering**, and **modeling pipelines** are fully implemented using:
- Pandas  
- NumPy  
- Matplotlib  

The project structure allows easy integration of new datasets with similar schemas.

---

## 🧪 Key Features

- Modular multi-phase architecture  
- Combination of classical ML and transformer-based tabular models  
- Integration of predictive analytics with LLM reasoning  
- Realistic insurance underwriting workflow  
- Production-oriented design  

---

## 🎯 Use Cases

- Fire insurance underwriting automation  
- Risk-aware policy evaluation  
- Customer churn prediction and retention planning  
- AI-assisted decision support for underwriters  

---

## 📌 Future Improvements

- Model explainability (SHAP, LIME)  
- Real-time API deployment  
- Human-in-the-loop underwriting  
- Multilingual LLM responses  
- Integration with core insurance systems  

---

## 📄 License

This project is provided for research and demonstration purposes.  
Commercial usage depends on internal company policies and data access permissions.
```
