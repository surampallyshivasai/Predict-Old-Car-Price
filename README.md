<<<<<<< HEAD
# Old Car Price Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-1.31-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" />
</p>

> An end-to-end Machine Learning project that predicts the **selling price of a used car** using regression algorithms. Includes a beautiful Streamlit web app for interactive predictions.

---

## 🗂️ Project Structure

```
Old Car Price Predection/
├── data/
│   └── car_data.csv          # Used car dataset
├── models/
│   ├── best_model.pkl        # Saved best ML model
│   ├── scaler.pkl            # Fitted StandardScaler
│   ├── label_encoders.pkl    # Fitted LabelEncoders
│   └── model_metadata.json   # Model performance metadata
├── notebooks/
│   ├── *.png                 # EDA + model evaluation plots
│   └── model_results.csv     # Model comparison table
├── src/
│   ├── preprocessing.py      # Data loading & feature engineering
│   ├── eda.py                # Exploratory Data Analysis
│   ├── train.py              # Model training & evaluation
│   └── predict.py            # CLI prediction interface
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 📊 Dataset Description

The dataset (`data/car_data.csv`) contains information on used cars with the following columns:

| Column          | Type        | Description                              |
|-----------------|-------------|------------------------------------------|
| `Car_Name`      | Categorical | Brand/model name of the car              |
| `Year`          | Integer     | Year of manufacture                      |
| `Selling_Price` | Float       | **Target** – Selling price in ₹ Lakhs    |
| `Present_Price` | Float       | Current ex-showroom price in ₹ Lakhs     |
| `Kms_Driven`    | Integer     | Total kilometres driven                  |
| `Fuel_Type`     | Categorical | Petrol / Diesel / CNG / Electric         |
| `Seller_Type`   | Categorical | Dealer / Individual                      |
| `Transmission`  | Categorical | Manual / Automatic                       |
| `Owner`         | Integer     | Number of previous owners (0–3)          |

---

## 🚀 Steps to Run the Project

### 1. Clone / Navigate to the project

```bash
cd "d:\Projects\Old Car Price Predection"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run EDA (Optional – generates plots in `notebooks/`)

```bash
python src/eda.py
```

### 4. Train the model

```bash
python src/train.py
```

This will:
- Preprocess the data
- Train & compare all models
- Save the best model, scaler, and encoders to `models/`
- Generate evaluation plots in `notebooks/`

### 5. Launch the Streamlit web app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 6. (Alternative) CLI Prediction

```bash
# Interactive mode
python src/predict.py

# Argument mode
python src/predict.py --year 2017 --present_price 9.85 --kms_driven 22000 \
  --fuel_type Diesel --seller_type Dealer --transmission Manual --owner 0
```

---

## 🤖 Model Details

Four regression models are trained and compared:

| Model               | Description                                       |
|---------------------|---------------------------------------------------|
| Linear Regression   | Baseline parametric model                         |
| Decision Tree       | Non-linear tree-based model (max_depth=6)         |
| Random Forest       | Ensemble of 200 trees (best performer typically) |
| XGBoost *(optional)*| Gradient boosting (requires `xgboost` package)   |

### Evaluation Metrics

| Metric | Description                              |
|--------|------------------------------------------|
| **R²** | Coefficient of determination (1.0 = perfect) |
| **MAE**| Mean Absolute Error – avg. prediction error  |
| **RMSE**| Root Mean Squared Error – penalises large errors |

The model with the highest **R² score** is automatically selected and saved.

---

## 🏁 Example Prediction

**Input:**

| Field           | Value     |
|-----------------|-----------|
| Year            | 2017      |
| Present Price   | ₹9.85 L   |
| Kms Driven      | 22,000    |
| Fuel Type       | Diesel    |
| Seller Type     | Dealer    |
| Transmission    | Manual    |
| Owner           | 0 (First) |

**Output:** `₹ 7.25 Lakhs` *(approximate – varies with model training)*

---

## 📈 EDA Highlights

- **Present Price** has the highest positive correlation with Selling Price
- **Car Age** (derived from Year) negatively correlates with price
- **Diesel** cars command higher resale value than Petrol
- **Individual sellers** tend to price lower than Dealers
- **Automatic** cars have higher resale prices on average

---

## 🛠️ Tech Stack

- **Language**: Python 3.9+
- **Data**: pandas, numpy
- **Visualisation**: matplotlib, seaborn
- **ML**: scikit-learn, xgboost
- **Serialisation**: joblib
- **Web App**: Streamlit

---

## 📄 License

This project is for **educational purposes** only.
=======
# Predict-Old-Car-Price
>>>>>>> 4a7fb483f56807f052c3c990811a02ec1f795b7c
