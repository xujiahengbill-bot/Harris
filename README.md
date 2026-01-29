# Titanic Survival Prediction – Ensemble Learning

This project predicts passenger survival on the Titanic using supervised
machine learning models including Logistic Regression, Random Forest,
Gradient Boosting, and XGBoost.
Features:
- Feature engineering (family size, isolation)
- Model comparison
- Stacking ensemble with XGBoost
- Cross-validation evaluation
Results:
- Best CV accuracy: ~0.84
- Final model: Stacking ensemble
Tools:
Python, pandas, scikit-learn, XGBoost

## Predicting Heart Disease Presence
This project builds a logistic regression model to predict whether a patient has heart disease using clinical and demographic data.
Dataset: 303 patients, binary outcome (Disease / No Disease)
Key Features: age, chest pain type, exercise-induced angina, ST depression (oldpeak), vessel blockage (ca), sex
Methods: exploratory data analysis, feature transformation, logistic regression with AIC/BIC stepwise selection
Performance: ~84% accuracy, ROC/AUC ≈ 0.92 on test data
Results show that exercise-induced angina, ST depression, and vessel blockage are strong risk indicators, while certain chest pain types and normal thal results are associated with lower disease risk. The final model balances interpretability and predictive power, making it suitable for clinical risk assessment.

## Sleep Health Data Analysis
This project analyzes sleep health across different occupations using the Sleephealth.csv dataset, focusing on sleep duration, sleep quality, stress levels, BMI, heart rate, and daily steps.
Methods: descriptive analysis, stepwise regression, ANOVA
Key Findings:
Sleep duration varies across occupations, with noticeable differences among doctors, teachers, nurses, engineers, and lawyers.
Quality of sleep is a strong and significant predictor of sleep duration.
Daily steps show a weak positive relationship with sleep duration; heart rate is not significant.
Higher BMI is associated with higher average stress, but ANOVA shows no statistically significant difference in stress across BMI categories.
Overall, the results highlight the importance of sleep quality and occupational factors in understanding sleep health and stress patterns.

## Retail Industry Time Series Analysis
This project models and forecasts retail industry jobs using time series data from 2003–present, including jobs, employees, layoffs, and spending.
Methods: ARIMA, ARIMAX (with exogenous variables), VAR
Key Steps: stationarity testing (ADF), seasonality analysis, model diagnostics, RMSE comparison
Best Model: ARIMAX, incorporating employees and spending as external regressors
Why ARIMAX: slightly higher RMSE than ARIMA but more interpretable and better captures economic drivers
Output: 12-month forecast for 2024 (exported as CSV)
The results show that including external economic variables improves interpretability and supports more realistic long-term forecasting of retail jobs.

## Predicting High Purchases & Customer Segmentation
This project identifies high-value customers and segments shoppers to support personalized marketing and revenue growth.
Methods: Logistic Regression (high vs. low spenders), K-Means clustering
Feature Engineering: loyalty score, high-volume buyer flag, expensive purchase indicator, discount sensitivity
Model Performance: Logistic Regression achieved 100% accuracy with AUC = 1.00
Segmentation: K-Means (K=3) identified
High-value premium buyers
Discount-sensitive low spenders
High-volume loyal buyers
Business Impact: Enables targeted promotions, loyalty programs, and pricing strategies for each customer segment
This analysis demonstrates how predictive modeling and clustering can drive data-driven marketing decisions.

## Predicting House Prices Using Machine Learning
This project compares Decision Tree and Random Forest models to predict house prices using the Ames Housing dataset.
Dataset: Ames Housing (Kaggle)
Preprocessing: median imputation for missing values, feature scaling
Models: Decision Tree (interpretable), Random Forest (ensemble)
Evaluation Metrics: RMSE, R²
Results
Decision Tree: RMSE = 38,602, R² = 0.8057
Random Forest: RMSE = 29,623, R² = 0.8856
Random Forest outperforms Decision Tree, demonstrating the strength of ensemble methods in capturing complex relationships. Key predictive features include OverallQual and GrLivArea. Future work includes hyperparameter tuning and exploring Gradient Boosting models.
