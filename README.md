# NUMERAI STOCK PREDICTION CHALLENGE
**Author: Tanmay Singh**

---

## Project Overview

This repository presents a meta-learning-based solution for the **NumerAI** tournament, involving the design and training of a **two-level stacked ensemble model**. The goal is to predict financial data patterns effectively, even in the presence of noise, imbalance, and rapidly changing distributions.

Only a subset of successful runs and model checkpoints are stored in this repository for backup.  
**Note:** The latest commit corresponds to the final submission, with a meta-test correlation of **0.018**.

---

## Key Contributions

- Developed a **stacked ensemble architecture** comprising 6 expert models and a meta-model (XGBoost, Random Forest, LightGBM, etc.).
- Applied **meta-learning techniques** with feature neutralization and **balanced sampling strategies** to address dataset drift and class imbalance.
- Boosted generalization by combining diverse models and fine-tuning evaluation using **meta-testing correlation** and feature-neutral metrics.

> **Tech Stack:** Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Imbalanced-learn, Git/GitHub

---

## Setup & Usage Instructions

### 1. Install Dependencies & Download Assets

```bash
python pre-installs.py
```

 Function of pre-installs.py:
- Downloads and installs all required libraries.
- Creates:
   - ```data```: for downloading NumerAI data via API.
   - ```saved_models```: to download pretrained models from Google Drive using ```gdown```.

### 2. Initialize Models

```bash
python Models/models.py
```

- Initializes all expert models and the meta-model architecture.

### 3. Train Models

```bash
# Jupyter
jupyter notebook train.ipynb

# Or Script
python train.py
```

- Trains models and saves new pickle files to ```saved_models```.
- To revert to using original Google Drive models, re-run ```pre-installs.py```.

### 4. Validate Trained Models

```bash
# Jupyter
jupyter notebook validation.ipynb

# Or Script
python validation.py
```

- Evaluates performance on the meta-testing (validation) parquet.

### 5. Generate Live Predictions

```bash
# Jupyter
jupyter notebook predict.ipynb

# Or Script
python predict.py
```

- Generates and saves predictions in a new ```predictions``` directory.

***IMPORTANT NOTE***

- Before every run, execute ```pre-installs.py``` to ensure a fresh slate on every iteration.
- If you're using newly trained models, comment out the block that downloads models from Google Drive using ```gdown```.

### Repository Structure

```bash
.
├── data/                # NumerAI dataset (auto-downloaded)
├── saved_models/        # Trained model pickles (auto-downloaded)
├── predictions/         # Stores output predictions
├── Models/
│   └── models.py        # Model architecture definitions
├── train.ipynb / .py    # Training pipeline
├── validation.ipynb/.py # Validation script
├── predict.ipynb/.py    # Prediction generation
└── pre-installs.py      # Setup script
```
