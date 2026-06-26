# Predictive Analytics - Marketing Sales Forecast

This project completes the Edutech Solution Data Analytics Internship Task 12.
It uses a marketing dataset and a Scikit-learn Linear Regression model to
forecast future sales outcomes from advertising spend.

## Project Files

- `data/marketing_dataset.csv` - generated marketing dataset used for training.
- `notebooks/marketing_sales_prediction.ipynb` - notebook version for review.
- `src/train_model.py` - complete training, evaluation, prediction, and reporting script.
- `models/linear_regression_model.pkl` - saved trained model file.
- `reports/prediction_report.md` - prediction report with metrics and interpretation.
- `reports/model_metrics.json` - machine-readable evaluation metrics.
- `reports/sample_predictions.csv` - actual vs predicted values on test data.
- `reports/future_predictions.csv` - forecasted outcomes for future marketing plans.
- `reports/actual_vs_predicted.png` - model performance visualization.

## Method

The model predicts `sales` from:

- `tv_ad_spend`
- `radio_ad_spend`
- `social_media_spend`
- `email_campaign_spend`
- `discount_percent`

The workflow follows the assignment hints:

1. Load or generate the marketing dataset.
2. Train a Linear Regression model.
3. Evaluate the model using RMSE, MAE, and R2 score.
4. Make future sales predictions.
5. Save the trained model and prediction report.

## How To Run

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the project:

```powershell
python src/train_model.py
```

The script recreates all generated files under `data/`, `models/`, and
`reports/`.

## Notebook

Open this notebook in Jupyter, VS Code, or GitHub:

```text
notebooks/marketing_sales_prediction.ipynb
```

The notebook contains the complete task workflow: data loading, Linear
Regression model training, evaluation, future predictions, and conclusions.

## Submission Checklist

- [x] Task PDF included.
- [x] Marketing dataset included.
- [x] Python source code included.
- [x] Jupyter notebook included.
- [x] Trained model file included.
- [x] Prediction report included.
- [x] Evaluation metrics included.
- [x] Future predictions included.
- [x] README explains the project and how to run it.

## Interview Questions

**Regression vs Classification:** Regression predicts continuous numeric values,
such as sales revenue. Classification predicts categories, such as whether a
customer will buy or not buy.

**What is RMSE?** RMSE means Root Mean Squared Error. It measures the average
prediction error in the same unit as the target variable. Lower RMSE means the
model predictions are closer to the actual values.
