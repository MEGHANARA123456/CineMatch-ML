# 🎬 Personalized Recommendation System

A machine learning-based recommendation engine that provides personalized content suggestions using collaborative filtering, content-based filtering, and a hybrid approach.

> Built as part of an enterprise-level Data Science & ML internship project.

---

## 📁 Project Structure

```
recommendation-system/
│
├── data/
│   ├── raw/                    # Original MovieLens dataset files
│   └── processed/              # Cleaned and preprocessed data
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_collaborative_filtering.ipynb
│   ├── 03_content_based.ipynb
│   ├── 04_hybrid_model.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│   ├── collaborative/
│   │   ├── user_based.py
│   │   ├── item_based.py
│   │   └── matrix_factorization.py
│   ├── content_based/
│   │   └── tfidf_recommender.py
│   ├── hybrid/
│   │   └── hybrid_recommender.py
│   └── utils/
│       ├── data_loader.py
│       ├── preprocessor.py
│       └── evaluator.py
│
├── models/                     # Saved .pkl model files
├── app/app.py                  # Streamlit dashboard
├── reports/figures/            # Charts and plots
├── tests/
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/recommendation-system.git
cd recommendation-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Get [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) and place files in `data/raw/`.

### 4. Run notebooks in order
```
01_EDA → 02_collaborative_filtering → 03_content_based → 04_hybrid_model → 05_evaluation
```

### 5. Launch Streamlit app
```bash
streamlit run app/app.py
```

---

## 📊 Models Implemented

| Model | Type | RMSE |
|---|---|---|
| User-based CF | Collaborative | ~1.02 |
| Item-based CF | Collaborative | ~0.98 |
| SVD | Matrix Factorization | ~0.93 |
| NMF | Matrix Factorization | ~0.96 |
| TF-IDF | Content-based | — |
| **Hybrid** | **Combined** | **~0.91** |

---

## 📈 Evaluation Metrics
- RMSE, MAE, Precision@K

## 🛠️ Tech Stack
Python · Pandas · Scikit-learn · Surprise · Streamlit · Matplotlib · Jupyter

---



## 📅 Timeline
- Week 1: Data + EDA | Week 2: CF Models | Week 3: Hybrid | Week 4: Dashboard + Report

## 📄 License
MIT License
