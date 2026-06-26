from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

DATA_PATH = DATA_DIR / "marketing_dataset.csv"
MODEL_PATH = MODEL_DIR / "linear_regression_model.pkl"
METRICS_PATH = REPORT_DIR / "model_metrics.json"
REPORT_PATH = REPORT_DIR / "prediction_report.md"
TEST_PREDICTIONS_PATH = REPORT_DIR / "sample_predictions.csv"
FUTURE_PREDICTIONS_PATH = REPORT_DIR / "future_predictions.csv"
CHART_PATH = REPORT_DIR / "actual_vs_predicted.png"

FEATURE_COLUMNS = [
    "tv_ad_spend",
    "radio_ad_spend",
    "social_media_spend",
    "email_campaign_spend",
    "discount_percent",
]
TARGET_COLUMN = "sales"


def ensure_directories() -> None:
    for directory in (DATA_DIR, MODEL_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def create_marketing_dataset(rows: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tv_ad_spend = rng.uniform(5_000, 80_000, rows).round(2)
    radio_ad_spend = rng.uniform(1_000, 25_000, rows).round(2)
    social_media_spend = rng.uniform(2_000, 45_000, rows).round(2)
    email_campaign_spend = rng.uniform(500, 12_000, rows).round(2)
    discount_percent = rng.uniform(0, 35, rows).round(2)

    seasonal_lift = rng.normal(1.0, 0.08, rows)
    noise = rng.normal(0, 3500, rows)

    sales = (
        15_000
        + tv_ad_spend * 0.055
        + radio_ad_spend * 0.11
        + social_media_spend * 0.16
        + email_campaign_spend * 0.22
        + discount_percent * 420
    ) * seasonal_lift + noise

    dataset = pd.DataFrame(
        {
            "campaign_id": [f"CMP-{number:03d}" for number in range(1, rows + 1)],
            "tv_ad_spend": tv_ad_spend,
            "radio_ad_spend": radio_ad_spend,
            "social_media_spend": social_media_spend,
            "email_campaign_spend": email_campaign_spend,
            "discount_percent": discount_percent,
            "sales": np.maximum(sales, 0).round(2),
        }
    )
    dataset.to_csv(DATA_PATH, index=False)
    return dataset


def load_or_create_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return create_marketing_dataset()


def train_model(dataset: pd.DataFrame) -> tuple[LinearRegression, pd.DataFrame, dict[str, float]]:
    x = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    prediction_table = x_test.copy()
    prediction_table["actual_sales"] = y_test.values
    prediction_table["predicted_sales"] = predictions.round(2)
    prediction_table["prediction_error"] = (
        prediction_table["actual_sales"] - prediction_table["predicted_sales"]
    ).round(2)
    prediction_table.to_csv(TEST_PREDICTIONS_PATH, index=False)

    metrics = {
        "rows": float(len(dataset)),
        "train_rows": float(len(x_train)),
        "test_rows": float(len(x_test)),
        "rmse": float(round(rmse, 2)),
        "mae": float(round(mae, 2)),
        "r2_score": float(round(r2, 4)),
        "intercept": float(round(model.intercept_, 2)),
    }
    metrics.update(
        {
            f"coefficient_{feature}": float(round(coefficient, 4))
            for feature, coefficient in zip(FEATURE_COLUMNS, model.coef_)
        }
    )

    return model, prediction_table, metrics


def make_future_predictions(model: LinearRegression) -> pd.DataFrame:
    future_campaigns = pd.DataFrame(
        [
            {
                "campaign_name": "Balanced Awareness Campaign",
                "tv_ad_spend": 45_000,
                "radio_ad_spend": 12_000,
                "social_media_spend": 22_000,
                "email_campaign_spend": 6_000,
                "discount_percent": 10,
            },
            {
                "campaign_name": "Digital Growth Campaign",
                "tv_ad_spend": 25_000,
                "radio_ad_spend": 8_000,
                "social_media_spend": 38_000,
                "email_campaign_spend": 9_000,
                "discount_percent": 15,
            },
            {
                "campaign_name": "Premium Launch Campaign",
                "tv_ad_spend": 70_000,
                "radio_ad_spend": 20_000,
                "social_media_spend": 35_000,
                "email_campaign_spend": 11_000,
                "discount_percent": 5,
            },
        ]
    )
    future_campaigns["predicted_sales"] = model.predict(
        future_campaigns[FEATURE_COLUMNS]
    ).round(2)
    future_campaigns.to_csv(FUTURE_PREDICTIONS_PATH, index=False)
    return future_campaigns


def save_chart(prediction_table: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(
        prediction_table["actual_sales"],
        prediction_table["predicted_sales"],
        color="#2563eb",
        alpha=0.75,
        edgecolors="#0f172a",
        linewidths=0.4,
    )
    lower = min(
        prediction_table["actual_sales"].min(),
        prediction_table["predicted_sales"].min(),
    )
    upper = max(
        prediction_table["actual_sales"].max(),
        prediction_table["predicted_sales"].max(),
    )
    plt.plot([lower, upper], [lower, upper], color="#dc2626", linewidth=2)
    plt.title("Actual vs Predicted Sales")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=160)
    plt.close()


def save_report(metrics: dict[str, float], future_predictions: pd.DataFrame) -> None:
    coefficients = [
        (feature, metrics[f"coefficient_{feature}"]) for feature in FEATURE_COLUMNS
    ]
    best_feature = max(coefficients, key=lambda item: abs(item[1]))

    future_rows = "\n".join(
        f"| {row.campaign_name} | {row.predicted_sales:,.2f} |"
        for row in future_predictions.itertuples(index=False)
    )

    report = f"""# Prediction Report

## Project Objective

The objective of this project is to forecast future sales outcomes using a
marketing dataset and Linear Regression. The model studies the relationship
between advertising spend, discount percentage, and sales.

## Dataset Summary

- Total records: {int(metrics["rows"])}
- Training records: {int(metrics["train_rows"])}
- Testing records: {int(metrics["test_rows"])}
- Target variable: `sales`
- Feature variables: `{ "`, `".join(FEATURE_COLUMNS) }`

## Model

- Algorithm: Scikit-learn Linear Regression
- Saved model file: `models/linear_regression_model.pkl`

## Evaluation Results

| Metric | Value |
| --- | ---: |
| RMSE | {metrics["rmse"]:,.2f} |
| MAE | {metrics["mae"]:,.2f} |
| R2 Score | {metrics["r2_score"]:.4f} |

RMSE measures the typical prediction error in the same unit as sales. The R2
score explains how much of the variation in sales is captured by the model.

## Feature Impact

| Feature | Coefficient |
| --- | ---: |
"""

    for feature, coefficient in coefficients:
        report += f"| {feature} | {coefficient:,.4f} |\n"

    report += f"""
The strongest coefficient in this trained model is `{best_feature[0]}`. This
means that, all else being equal, changes in this feature have the largest
estimated effect on predicted sales.

## Future Predictions

| Campaign | Predicted Sales |
| --- | ---: |
{future_rows}

## Conclusion

The Linear Regression model can forecast future marketing outcomes and estimate
how advertising channels influence sales. The saved model file can be reused to
predict sales for new campaign budgets.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_directories()
    dataset = load_or_create_dataset()
    model, prediction_table, metrics = train_model(dataset)
    future_predictions = make_future_predictions(model)

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_chart(prediction_table)
    save_report(metrics, future_predictions)

    print(f"Dataset saved to: {DATA_PATH}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Report saved to: {REPORT_PATH}")
    print(f"RMSE: {metrics['rmse']:,.2f}")
    print(f"R2 Score: {metrics['r2_score']:.4f}")


if __name__ == "__main__":
    main()
