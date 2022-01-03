# M5 Forecasting Kaggle

The Makridakis Open Forecasting Center (MOFC) at the University of Nicosia conducts cutting-edge forecasting research and provides business forecast training. It helps companies achieve accurate predictions, estimate the levels of uncertainty, avoiding costly mistakes, and apply best forecasting practices. The MOFC is well known for its Makridakis Competitions, the first of which ran in the 1980s.  

In this competition, the fifth iteration, we will use hierarchical sales data from Walmart, the world’s largest company by revenue, to forecast daily sales for the next 28 days. The data, covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. In addition, it has explanatory variables such as price, promotions, day of the week, and special events. Together, this robust dataset can be used to improve forecasting accuracy.

## Features
The feature engineering is mainly general date features and aggregations of lagged sales at varied levels to reduce uncertainty. For the aggregated features, levels of aggregation were :

- Item and store
- Item (aggregated over all stores)
- Dept id and store id

## Modelling
- Starter scripts of lightgbm, XGB etc. to establish benchmark of the performance
- RNN, LSTM scripts to check for improvement in accuracy

