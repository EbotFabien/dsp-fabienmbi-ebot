# 🏠 Ames Housing Price Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.2-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview
This project predicts **house sale prices** in Ames, Iowa, using a subset of features from the Ames Housing dataset.  
The purpose is to demonstrate **data preprocessing, feature scaling, encoding, and regression modeling** in Python with `pandas`, `numpy`, and `scikit-learn`.  

We focus on **4 features** for this demonstration:

- **Continuous Features:**  
  - `GrLivArea` – Above grade living area (sq. ft.)  
  - `YearBuilt` – Year the house was built  

- **Categorical Features:**  
  - `Neighborhood` – Nominal neighborhood location  
  - `KitchenQual` – Ordinal kitchen quality (`Po < Fa < TA < Gd < Ex`)  

---

## Project Structure

```text
project/
│
├── data/
│   └── dataset.csv          # Ames Housing dataset
│
├── scripts/
│   ├── preprocess.py        # Manual preprocessing of features
│   ├── train_model.py       # Model training and evaluation
│   └── utils.py             # Helper functions (e.g., RMSLE)
│
├── requirements.txt         # Python dependencies
└── README.md
```

## Dependencies

**Required Python packages:**

```text
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
```
## How to Run

1. **Clone the repository**

```bash
git clone https://github.com/EbotFabien/dsp-fabienmbi-ebot.git
cd dsp-fabienmbi-ebot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Run notebook**
```bash
cd notebooks
jupter notebook
```

