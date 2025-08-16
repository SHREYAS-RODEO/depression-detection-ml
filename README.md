# Depression Detection from Reddit Posts (NLP + ML)

NLP pipeline to classify Reddit posts as depressive vs non-depressive using two approaches:
- **Classical ML**: TF-IDF + Linear SVM
- **Transformer**: BERT fine-tuning (HuggingFace)

> **Note:** Research/education only. Not a clinical diagnostic tool.

---

## 📊 Performance Summary (fill these with your run results)

- **Best Model:** `TODO: e.g., BERT (bert-base-uncased)` or `LinearSVC + TF-IDF`
- **Accuracy:** `TODO`
- **Precision / Recall / F1:** `TODO`
- **ROC-AUC:** `TODO`
- **Statistical test:** `TODO: t-test p-value comparing BERT vs SVM correctness`

> Put the actual numbers your notebook prints (classification report, ROC, and t-test).

---

## 📁 Project Structure
├── dep_det.ipynb # Main notebook
├── requirements.txt # Dependencies to run the notebook
└── README.md # This file


---

## 📊 Dataset

- Expects a CSV named **`reddit_depression_dataset.csv`** in the working directory.
- Required columns: **`body`** (text) and **`label`** (0/1).
- The notebook drops missing values, casts labels to `int`, and **samples ~1,000 rows** for a quick baseline.

> The dataset is **not included** in this repo due to size and licensing. Add your own copy locally.

---

## 🧠 Method

**Preprocessing**
- Regex-based text cleaning (URLs, non-letters, digits), lowercasing
- Optional stopword removal / lemmatization (as configured in the notebook)
- WordCloud visualizations (per class)

**Features**
- TF-IDF (uni/bi-grams, `max_features=5000`)
- Optional PCA for analysis

**Models**
- **LinearSVC** (scikit-learn)
- **BERT** (`bert-base-uncased`) fine-tuned with HuggingFace `Trainer`

**Evaluation**
- Train/test split (stratified)
- Metrics: Accuracy, Precision, Recall, F1
- Confusion matrix (SVM)
- ROC and Precision-Recall curves (SVM and BERT)
- **Two-sample t-test** on per-example correctness (BERT vs SVM)

---

## ✅ Results (example layout — replace with your real numbers)

| Model               | Accuracy | Precision | Recall | F1   | ROC-AUC |
|---------------------|---------:|----------:|------:|-----:|--------:|
| Linear SVM (TF-IDF) |   `TODO` |    `TODO` | `TODO`|`TODO`|  `TODO` |
| BERT (fine-tuned)   |   `TODO` |    `TODO` | `TODO`|`TODO`|  `TODO` |

- **t-test (BERT vs SVM) p-value:** `TODO` (if < 0.05, the difference is statistically significant)
- **Takeaway:** `TODO: one-line insight (e.g., BERT > SVM on this sample; TF-IDF+SVM remains a strong baseline).`

---

## ⚙️ Setup & Run

1) **Create environment & install deps**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
