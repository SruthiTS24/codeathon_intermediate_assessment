# codeathon_intermediate_assessment
# Car Price Prediction — Multiple Regression Models

## Overview
A machine learning project to identify significant factors affecting car prices in the American market and build a predictive model for price estimation. Built for an automobile consulting firm assisting a Chinese manufacturer planning US market entry.

---

## Dataset
- **Source:** CarPrice_Assignment.csv
- **Size:** 205 rows × 26 columns
- **Target Variable:** `price`

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## Project Pipeline

### 1. Preprocessing
- Dropped `car_ID` (serial number, no predictive value)
- Extracted `CarBrand` from `CarName` and corrected 5 typos (`vw`, `vokswagen`, `maxda`, `porcshce`, `toyouta`)
- Converted text-based numeric columns (`doornumber`, `cylindernumber`) to integers
- Binary encoded `fueltype`, `aspiration`, `enginelocation` using explicit mapping
- One-hot encoded `carbody`, `drivewheel`, `enginetype`, `fuelsystem`, `CarBrand` with `drop_first=True`
- **Final shape after preprocessing:** 205 × 60, fully numeric, zero nulls

### 2. Models Implemented
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)

### 3. Evaluation Metrics
R² Score, Mean Squared Error (MSE), Mean Absolute Error (MAE)

---

## Results

| Model | R² | MSE | MAE |
|---|---|---|---|
| **Random Forest** | **0.9577** | **3,343,086.97** | **1,285.09** |
| Gradient Boosting | 0.9281 | 5,678,838.06 | 1,681.26 |
| Decision Tree | 0.9022 | 7,722,717.65 | 1,935.27 |
| Linear Regression | 0.8951 | 8,283,565.72 | 1,942.07 |
| SVR | -0.1008 | 86,898,072.68 | 5,701.89 |

**Best Model: Random Forest Regressor** — highest R² and lowest MAE across all models.

---

## 4. Feature Importance
Top predictors identified from Random Forest feature importance analysis:

| Feature | Importance Score |
|---|---|
| enginesize | 0.5465 |
| curbweight | 0.2985 |
| highwaympg | 0.0458 |
| horsepower | 0.0319 |
| carwidth | 0.0131 |

> Engine size and curb weight together account for ~84% of total feature importance.

---

## 5. Hyperparameter Tuning
GridSearchCV with 5-fold cross-validation applied to Random Forest (108 combinations, 540 fits).

**Best Parameters:**
```python
n_estimators=300, max_depth=10, min_samples_split=2, min_samples_leaf=1
```

| Model | R² | MAE |
|---|---|---|
| Baseline | 0.9577 | 1,285.09 |
| Tuned | 0.9585 | 1,222.09 |

---

## Conclusion
Random Forest Regressor with tuned hyperparameters is the optimal model for this problem. Engine size and curb weight are the primary pricing determinants in the American market, providing actionable insights for vehicle design and pricing strategy.

---

## Repository Structure
```
├── CarPrice_Assignment.csv
├── Codeathon_Intermediate_Assessment.ipynb
└── README.md
```
