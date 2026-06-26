# Prediction Report

## Project Objective

The objective of this project is to forecast future sales outcomes using a
marketing dataset and Linear Regression. The model studies the relationship
between advertising spend, discount percentage, and sales.

## Dataset Summary

- Total records: 240
- Training records: 192
- Testing records: 48
- Target variable: `sales`
- Feature variables: `tv_ad_spend`, `radio_ad_spend`, `social_media_spend`, `email_campaign_spend`, `discount_percent`

## Model

- Algorithm: Scikit-learn Linear Regression
- Saved model file: `models/linear_regression_model.pkl`

## Evaluation Results

| Metric | Value |
| --- | ---: |
| RMSE | 4,309.86 |
| MAE | 3,513.86 |
| R2 Score | 0.5571 |

RMSE measures the typical prediction error in the same unit as sales. The R2
score explains how much of the variation in sales is captured by the model.

## Feature Impact

| Feature | Coefficient |
| --- | ---: |
| tv_ad_spend | 0.0711 |
| radio_ad_spend | 0.0499 |
| social_media_spend | 0.1274 |
| email_campaign_spend | 0.0519 |
| discount_percent | 365.3478 |

The strongest coefficient in this trained model is `discount_percent`. This
means that, all else being equal, changes in this feature have the largest
estimated effect on predicted sales.

## Future Predictions

| Campaign | Predicted Sales |
| --- | ---: |
| Balanced Awareness Campaign | 27,970.61 |
| Digital Growth Campaign | 30,369.79 |
| Premium Launch Campaign | 30,237.79 |

## Conclusion

The Linear Regression model can forecast future marketing outcomes and estimate
how advertising channels influence sales. The saved model file can be reused to
predict sales for new campaign budgets.
