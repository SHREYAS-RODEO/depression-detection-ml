# Depression Detection from Reddit Posts (NLP + ML)

This project classifies Reddit posts as depressive vs non-depressive using:
- **Classical ML models**: SVM, Logistic Regression, Random Forest, Naive Bayes
- **Transformer model**: BERT (fine-tuned with HuggingFace)

> **Note:** This project is for research and educational purposes only.  
> It is **not** a clinical diagnostic tool.

---

## 📊 Performance Summary

- **Best Model:** BERT (bert-base-uncased)  
- **Accuracy:** 86.1%  
- **Precision:** 87%  
- **Recall:** 85%  
- **F1-score:** 86%  
- **ROC-AUC:** 0.93  

BERT outperformed all classical algorithms across metrics.  
SVM followed closely with **85.3% accuracy** and **0.92 AUC**, showing that classical TF-IDF + SVM remains a strong baseline. Logistic Regression and Random Forest also achieved competitive performance.

---

## 📁 Project Structure

├── dep_det.ipynb # Main notebook with training + evaluation
├── requirements.txt # Dependencies
└── README.md # Documentation


---

## 📊 Dataset

- Input: `reddit_depression_dataset.csv` (not included in repo due to size/licensing)  
- Required columns:  
  - **body** → text of Reddit post  
  - **label** → 0 (non-depressive), 1 (depressive)  
- Notebook drops missing values and samples ~1,000 rows for a quick baseline.

---

## 🧠 Methodology

### Preprocessing
- Regex-based cleaning (remove URLs, digits, special chars)  
- Lowercasing  
- Stopword removal & lemmatization  
- WordClouds to visualize frequent terms per class  

### Feature Extraction
- **TF-IDF Vectorizer** (`max_features=5000`)  
- **PCA** (optional) for dimensionality reduction  

### Models
- **Classical ML:** SVM, Logistic Regression, Random Forest, Naive Bayes  
- **Deep Learning:** BERT (HuggingFace Transformers)  

### Evaluation
- Stratified train/test split  
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC  
- Confusion matrix (for SVM)  
- ROC/PR curves (SVM vs BERT)  
- Statistical comparison: two-sample t-test on per-example correctness (BERT vs SVM)  

---

## ✅ Results

| Model               | Accuracy | Precision | Recall | F1   | AUC  |
|---------------------|---------:|----------:|------:|-----:|-----:|
| SVM                 | 85.3%    | 86%       | 84%   | 85%  | 0.92 |
| Logistic Regression | 84.9%    | 85%       | 83%   | 84%  | 0.91 |
| Random Forest       | 83.4%    | 84%       | 82%   | 83%  | 0.89 |
| Naive Bayes         | 82.1%    | 83%       | 80%   | 81%  | 0.87 |
| **BERT**            | **86.1%**| **87%**   | **85%** | **86%** | **0.93** |

**Insight:**  
- BERT achieved the highest performance across all metrics.  
- SVM remained a competitive baseline with only slight degradation.  
- Logistic Regression and Random Forest provide viable alternatives under constrained resources.  

---

## ⚙️ Setup

1. Clone the repository:
```bash
git clone https://github.com/SHREYAS-RODEO/depression-detection-ml.git
cd depression-detection-ml
