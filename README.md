# Loan Data Analysis & Prediction

This project explores a loan approval dataset and performs exploratory data analysis (EDA), feature analysis, and predictive modeling to understand which factors influence loan status.

The analysis is implemented in the notebook:
- `DataAnalyse_1.ipynb`

## Project Overview

The dataset contains customer and loan attributes such as age, income, education, loan amount, interest rate, credit score, and prior default history. The goal is to analyze patterns in the data and build a model to classify whether a loan is approved or rejected (`loan_status`).

## Objectives

- Load and inspect the loan dataset
- Clean and validate the data
- Explore key relationships between features and loan outcome
- Visualize distributions and patterns using pandas, seaborn, and matplotlib
- Encode categorical variables and prepare the dataset for modeling
- Train and compare multiple classifiers
- Evaluate model performance using classification metrics

## Dataset

The dataset is loaded from an Excel file:
- `loan_data.xlsx`

Key columns include:
- `age`
- `gender`
- `education`
- `income`
- `person_emp_exp`
- `home_ownership`
- `loan_amount`
- `loan_intent`
- `loan_int_rate`
- `loan_percent_of_income`
- `cb_person_cred_hist_length`
- `credit_score`
- `previous_loan_defaults_on_file`
- `loan_status`

## Tools and Libraries

This project uses:
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Optuna
- SciPy

## Installation

1. Clone the repository.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
   ```
3. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna scipy openpyxl
   ```

## How to Run

Open the notebook in Jupyter or VS Code and run all cells:

```bash
jupyter notebook "DataAnalyse_1.ipynb"
```

Or use Jupyter Lab:

```bash
jupyter lab
```

## Project Workflow

1. Import libraries and read the Excel dataset
2. Inspect the schema and sample rows
3. Check missing values and basic statistics
4. Visualize distributions and relationships
5. Group and compare features by `loan_intent` and `loan_status`
6. Build engineered features such as income groups
7. Prepare data for model training
8. Apply preprocessing and model training
9. Evaluate model accuracy, precision, recall, and F1-score

## Notable Findings

The notebook includes analysis of:
- income vs. age and loan status
- loan amount and income by approval status
- differences in loan intent distribution
- impact of prior loan defaults on loan outcomes
- risk patterns across income groups
- variable distributions and outliers

## Modeling Approach

The notebook experiments with multiple classification models, including:
- Logistic-style linear modeling
- Decision Trees
- Random Forest
- AdaBoost
- Gradient Boosting
- SVM
- XGBoost
- Ensemble methods such as Voting and Stacking

The project also uses cross-validation and hyperparameter tuning concepts to improve model evaluation.

## Example Model Evaluation Metrics

The notebook calculates standard binary classification metrics, including:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Repository Structure

```text
data-analysis/
├── DataAnalyse_1.ipynb
├── loan_data.xlsx
├── README.md
```

## Future Improvements

- Add more robust feature engineering
- Handle class imbalance using SMOTE or class weighting
- Compare more advanced models and tuning pipelines
- Save the best model with joblib or pickle
- Deploy the prediction pipeline as an API or web application

## Author

This project was created for exploratory data analysis and machine learning experimentation on loan approval data.

## Notes

This notebook is intended for educational and analytical use. Some cells include plotting, experimentation, and model comparison tasks and may be adapted depending on the environment and dependencies installed.
