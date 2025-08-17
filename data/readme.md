# Used Car Price Prediction

## Project Overview
This project aims to build predictive models for estimating the prices of used cars in the Australian market. Various machine learning algorithms were applied, including Linear Regression, Decision Tree, Random Forest, and XGBoost. The dataset was cleaned, preprocessed, and analyzed to select the most relevant features.

## Dataset
- Source: Austrilia car market platforms
- Main features: Brand, Transmission, FuelType, Car_Age, Mileage, etc.
- Target variable: Price

## Methodology
1. Data cleaning and preprocessing (handling missing values, outlier removal).
2. Feature engineering (OneHotEncoding for categorical features, scaling numerical features).
3. Train-test split (80% training, 20% testing).
4. Model training: Linear Regression, Decision Tree, Random Forest, XGBoost.
5. Hyperparameter tuning with GridSearchCV and RandomizedSearchCV.
6. Model evaluation using MAE, RMSE, and R².

## Results
- Random Forest and XGBoost achieved the best performance after tuning.
- Important hyperparameters: `n_estimators`, `max_depth`, and `learning_rate` (for boosting models).

## Limitations
- Dataset size may limit model generalization.
- Price distribution is skewed and may bias predictions.
- Only structured data was used; additional features (e.g., text descriptions, images) could improve accuracy.

## Future Work
- Explore advanced hyperparameter optimization (Bayesian Optimization, Hyperband).
- Incorporate additional features (e.g., seller type, car photos).
- Deploy the model as a web service or application.

## Requirements
- Python 3.9+
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost

To install dependencies:
```bash
pip install -r requirements.txt

