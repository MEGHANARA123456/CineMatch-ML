# 🎬 CineMatch-ML: Personalized Movie Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-23%2F23%20Passing-brightgreen)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

> An enterprise-level, ML-powered movie recommendation system combining collaborative filtering, content-based filtering, and a hybrid approach — delivered through an interactive Streamlit dashboard.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Algorithms](#-algorithms)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Dependencies](#-dependencies)
- [Testing](#-testing)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)
- [License](#-license)

---

## 🧠 Overview

CineMatch-ML solves the problem of generic, one-size-fits-all movie recommendations by combining **5 different ML algorithms** into a tunable hybrid system. Users get personalized recommendations based on their rating history, while also discovering new content through genre similarity.

**Dataset:** MovieLens 100K — 100,000 ratings from 943 users on 1,682 movies.

---

## ✨ Features

### Core
- 5 recommendation algorithms: User-CF, Item-CF, SVD, NMF, TF-IDF, and Hybrid
- Tunable hybrid model with adjustable alpha (CF vs Content weighting)
- Real-time recommendations via Streamlit dashboard
- Model comparison with performance metrics (RMSE, MAE, Precision@K, Recall@K)
- Trending movies with composite popularity scoring
- Export recommendations as CSV

### UI/UX
- Dark / Light theme toggle
- Color-coded recommendation scores (🟢 ≥70%, 🟡 ≥50%, 🔴 <50%)
- Interactive sliders for parameter tuning
- Responsive two-column layout
- Random user selection for quick testing

---

## 🖥️ Demo

```bash
streamlit run app/app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 🏗️ Architecture

```
CineMatch-ML/
├── app/
│   └── app.py                      # Streamlit web application
├── src/
│   ├── collaborative/
│   │   ├── user_based.py           # User-based Collaborative Filtering
│   │   ├── item_based.py           # Item-based Collaborative Filtering
│   │   └── matrix_factorization.py # SVD & NMF
│   ├── content_based/
│   │   └── tfidf_recommender.py    # TF-IDF Content Filtering
│   ├── hybrid/
│   │   └── hybrid_recommender.py   # Weighted Hybrid Model
│   └── utils/
│       ├── data_loader.py          # Data loading utilities
│       ├── preprocessor.py         # Data cleaning & preprocessing
│       └── evaluator.py            # Performance metrics
├── models/                         # Serialized ML models (joblib)
├── data/
│   ├── raw/                        # Original MovieLens dataset
│   └── processed/                  # Cleaned datasets
├── notebooks/                      # Jupyter notebooks for development
├── tests/                          # Unit tests (23 test cases)
└── reports/                        # Evaluation results & figures
```

---

## 🤖 Algorithms

### Collaborative Filtering

| Algorithm | Description |
|-----------|-------------|
| **User-Based CF** | Finds similar users (cosine similarity, top-20) and recommends what they liked |
| **Item-Based CF** | Finds similar items based on user preferences — more scalable for new users |
| **SVD** | Matrix factorization with 100 latent factors, 20 epochs, lr=0.005 |
| **NMF** | Non-negative Matrix Factorization with 15 factors for interpretability |

### Content-Based Filtering

**TF-IDF Recommender** — Converts genre binary vectors to text, applies TF-IDF vectorization, and computes cosine similarity to recommend movies with similar genre profiles.

### Hybrid Model

```
final_score = α × CF_score + (1 - α) × content_score
```

Default α = 0.7 (70% collaborative, 30% content). Fully configurable via the dashboard slider.

---

## 📊 Model Performance

| Model | RMSE | MAE | Precision@10 | Recall@10 |
|-------|------|-----|--------------|-----------|
| User-CF | 1.02 | 0.81 | 0.983 | 0.800 |
| Item-CF | 0.98 | 0.78 | 0.983 | 0.800 |
| SVD | 0.94 | 0.74 | 0.716 | 0.547 |
| NMF | 0.96 | 0.76 | 0.698 | 0.519 |
| **Hybrid** ⭐ | **0.91** | **0.72** | **0.716** | **0.547** |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- Anaconda (recommended)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/MEGHANARA123456/CineMatch-ML.git
cd CineMatch-ML

# 2. Create and activate virtual environment
conda create -n recommender python=3.10
conda activate recommender

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download MovieLens 100K dataset and place in data/raw/
#    https://grouplens.org/datasets/movielens/100k/
```

---

## 🚀 Usage

### Run the Dashboard

```bash
streamlit run app/app.py
```

### Train Models

```bash
python src/train.py
```

### Evaluate Models

```bash
python src/utils/evaluator.py
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 📦 Dependencies

```
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
scikit-surprise==1.1.3
streamlit==1.28.0
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2
scipy==1.11.2
nltk==3.8.1
tqdm==4.66.1
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🧪 Testing

```bash
pytest tests/ -v --tb=short
```

- **23 test cases** across all modules
- **100% code coverage**
- Includes unit, integration, and performance tests

---

## 🔮 Future Enhancements

- User registration and persistent profiles
- Social features (share recommendations)
- Advanced filtering (genre, year, rating range)
- Real-time learning from user feedback
- Integration with IMDb / TMDb APIs
- REST API endpoints for third-party integration
- Mobile app

---

## 👩‍💻 Author

**Meghana Kamatam**  
GitHub: [@MEGHANARA123456](https://github.com/MEGHANARA123456)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).