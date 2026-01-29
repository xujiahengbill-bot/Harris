# Titanic Survival Prediction – Ensemble Learning

This project predicts passenger survival on the Titanic using supervised
machine learning models including Logistic Regression, Random Forest,
Gradient Boosting, and XGBoost.

## Features
- Feature engineering (family size, isolation)
- Model comparison
- Stacking ensemble with XGBoost
- Cross-validation evaluation

## Results
- Best CV accuracy: ~0.84
- Final model: Stacking ensemble

## Tools
Python, pandas, scikit-learn, XGBoost

## Predicting Heart Disease Presence
This project builds a logistic regression model to predict whether a patient has heart disease using clinical and demographic data.
Dataset: 303 patients, binary outcome (Disease / No Disease)
Key Features: age, chest pain type, exercise-induced angina, ST depression (oldpeak), vessel blockage (ca), sex
Methods: exploratory data analysis, feature transformation, logistic regression with AIC/BIC stepwise selection
Performance: ~84% accuracy, ROC/AUC ≈ 0.92 on test data
Results show that exercise-induced angina, ST depression, and vessel blockage are strong risk indicators, while certain chest pain types and normal thal results are associated with lower disease risk. The final model balances interpretability and predictive power, making it suitable for clinical risk assessment.
