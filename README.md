# Credit Card Fraud Detection

A Streamlit web application that uses a trained Random Forest Classifier to predict whether a credit card transaction is legitimate or fraudulent.

## Features

- Real-time fraud prediction with probability scores
- Interactive UI with 30 transaction feature inputs (Time, V1–V28, Amount)
- Pre-loaded sample transactions (normal and fraudulent) for quick testing
- Visual confidence indicator

## Model

| Detail | Value |
|--------|-------|
| Algorithm | Random Forest Classifier |
| Dataset | [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Features | 30 (Time, V1–V28 PCA components, Amount) |
| Preprocessing | StandardScaler on Time & Amount |

## Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

## Project Structure

```
.
├── app.py                    # Streamlit application
├── fraud_detection_model.pkl # Trained Random Forest model
├── fraud_scaler.pkl          # Fitted StandardScaler
├── feature_columns.pkl       # Ordered feature column list
├── requirements.txt
└── README.md
```

## Usage

1. Open the app in your browser (default: `http://localhost:8501`).
2. Optionally load a **sample transaction** from the dropdown for quick testing.
3. Adjust the feature values as needed.
4. Click **Predict** to see the fraud probability and verdict.
