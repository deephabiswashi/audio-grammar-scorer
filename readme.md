# Grammar Scoring Engine

A comprehensive framework for automatically scoring spoken‑audio grammar on a 0–5 MOS Likert scale. The repository contains two Kaggle‑style Jupyter notebooks:

* **`grammar-scoring-system-cnn.ipynb`**
  Implements a convolutional neural network on spectrogram inputs for grammar scoring, including feature extraction, model training, and performance visualization.

* **`grammar-scoring-system-for spoken-audio.ipynb`**
  Builds a feature‑based pipeline using Librosa‑extracted acoustic features and traditional regressors (Random Forest, XGBoost, simple NN), complete with RMSE evaluation and error analysis plots.

---

## 📂 Repository Structure

```
/
├── grammar-scoring-system-cnn.ipynb
├── grammar-scoring-system-for spoken-audio.ipynb
├── data/                  # Audio files and CSV labels
│   ├── audios/train/
│   ├── audios/test/
│   ├── train.csv
│   └── test.csv
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/deephabiswashi/GrammarScoringEngine.git
cd GrammarScoringEngine
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Place your `.wav` files in `data/audios/train/` and `data/audios/test/`, and ensure `train.csv` and `test.csv` are in `data/`.

### 4. Run Notebooks

Open and execute the notebooks in sequence:

1. `grammar-scoring-system-cnn.ipynb` for the CNN‑based approach.
2. `grammar-scoring-system-for spoken-audio.ipynb` for the feature‑based pipeline.

---

## 📋 Requirements

* Python 3.8+
* `librosa`, `pandas`, `numpy`, `scikit-learn`, `torch`, `tensorflow` (for NN variants), `xgboost`, `matplotlib`, `seaborn`, `transformers` (if using advanced approach)

Install via:

```bash
pip install librosa pandas numpy scikit-learn torch tensorflow xgboost matplotlib seaborn transformers
```

---

