# 🏠 House Price Prediction

An end-to-end machine learning project that predicts house prices from real estate listing data — covering data cleaning, exploratory data analysis, model training/comparison, and deployment via Flask and Streamlit.

## 📊 Project Overview

Real estate listing data is messy: prices are written as `"42 Lac"` or `"1.4 Cr"`, areas as `"1200 sqft"`, floors as `"4 out of 12"`. This project parses all of that into clean numeric/categorical features, explores the data, and trains multiple regression models to predict price — then serves the best model through a web app.

## 🗂 Project Structure

```
House Price Prediction/
├── .github/
│   └── workflows/              # CI/CD workflows
├── artifacts/                  # Generated at runtime (gitignored, tracked with DVC)
│   ├── raw.csv
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
│   └── preprocessor.pkl
├── notebook/
│   └── data/
│       └── house_prices.csv    # Raw dataset
├── EDA.ipynb                    # Data cleaning + exploratory data analysis
├── Model_Training.ipynb         # Model comparison (prototyping)
├── src/
│   └── ML_Project/
│       ├── components/
│       │   ├── data_ingestion.py        # Reads raw CSV, splits train/test
│       │   ├── data_transformation.py   # Feature engineering + preprocessing pipeline
│       │   └── model_trainer.py         # Trains & compares models, saves best one
│       ├── pipelines/          # Prediction / training pipeline orchestration
│       ├── logger.py
│       ├── exception.py
│       └── utils.py
├── Streamlit_app.py             # Streamlit web app
├── setup.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🧹 Data Cleaning & Feature Engineering

- Parsed `Amount(in rupees)` (`"42 Lac"` / `"1.4 Cr"`) into numeric rupee values
- Extracted area from Carpet/Super Area text fields
- Extracted `BHK` from listing titles
- Split `Floor` into `floor_num` and `total_floors`
- Cleaned `Bathroom` / `Balcony` / `Car Parking` fields
- Grouped low-frequency locations into `"other"` to control categorical cardinality
- Removed unrealistic outliers (extreme prices/areas) and applied a log transform to the right-skewed target

## 🤖 Models Trained

| Model | R² | MAE (log scale) | RMSE (log scale) |
|---|---|---|---|
| Linear Regression | 0.721 | 0.318 | 0.423 |
| Random Forest | 0.918 | 0.117 | 0.229 |
| XGBoost | 0.925 | 0.119 | 0.219 |
| CatBoost | 0.921 | 0.127 | 0.226 |
| **LightGBM (best)** | **0.929** | **0.109** | **0.214** |

Hyperparameters were tuned with `RandomizedSearchCV`. **LightGBM** was selected as the production model for the best balance of accuracy and error.

## 🚀 How to Run

### 1. Set up the environment
```bash
python -m venv venvr
venvr\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Run the training pipeline
```bash
python -m src.ML_Project.components.data_ingestion
```
This runs ingestion → transformation → training in sequence, and saves `model.pkl` and `preprocessor.pkl` into `artifacts/`.

### 3. Launch the web app

**Streamlit:**
```bash
streamlit run Streamlit_app.py
```

**Flask:**
```bash
python app.py
```

## 🛠 Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `LightGBM` · `XGBoost` · `CatBoost` · `Flask` · `Streamlit` · `DVC` · `GitHub Actions`

## 📦 Model & Data Versioning

Large artifacts (`model.pkl`, `preprocessor.pkl`, and training CSVs) are excluded from Git and version-controlled with [DVC](https://dvc.org/) instead, due to GitHub's file size limits.

```bash
dvc pull   # fetch model/data artifacts after cloning
dvc push   # push updated artifacts to remote storage
```

## 📄 License

This project is open-source and available for learning/portfolio purposes.