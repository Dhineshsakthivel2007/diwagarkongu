# 📊 AI-Based Retail Demand Forecasting System
Live demo : https://diwagarkongu-wrzjnjyoybrr2ff5clufmw.streamlit.app
## 🏆 Hackathon Project – Short-Term Product Demand Prediction

Retailers and supply chain organizations need accurate demand forecasting to manage inventory, reduce waste, and avoid revenue loss caused by overstocking or understocking.

This project builds an AI-powered demand forecasting system using historical sales time-series data.

---

## 🎯 Problem Statement

Design and develop an AI-based system that forecasts short-term product demand using historical sales data.

The system should:

- Identify trends and seasonality in sales patterns  
- Predict demand for upcoming time periods  
- Support inventory planning decisions  
- Visualize historical vs forecasted demand  

---

## ✅ Solution Overview

We implemented a demand forecasting model using **Facebook Prophet**, a time-series forecasting tool designed for business demand prediction.

The system provides:

- Future demand forecasts (next 30 days)
- Forecast visualization graphs
- Model evaluation metrics (MAE, RMSE, MAPE)
- Inventory planning recommendations (Safety Stock + Reorder Point)

---

## 🛠 Technologies Used

- Python  
- Pandas, NumPy  
- Matplotlib  
- Prophet (Time-Series Forecasting Model)  
- Scikit-learn (Evaluation Metrics)  
- Streamlit (Interactive Dashboard)

---

## 📂 Input Dataset Format

CSV file containing:

| Column       | Description              |
|------------|--------------------------|
| Date        | Sales date               |
| Product ID  | Product identifier       |
| Units Sold  | Quantity sold per day    |

Example:

```csv
Date,Product ID,Units Sold
2023-01-01,P001,120
2023-01-02,P001,135
2023-01-03,P001,142
```
⚙️ How It Works

Load historical sales time-series data

Select a product and aggregate daily demand

Train Prophet forecasting model

Predict demand for the next 30 days

Evaluate accuracy using MAE, RMSE, MAPE

Generate inventory recommendations

Display results using Streamlit dashboard
📉 Model Evaluation Metrics

We evaluate forecasting performance using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

MAPE (Mean Absolute Percentage Error)

MAPE is most useful for retail demand forecasting:

<10% → Excellent

10–20% → Good

20% → Needs improvement
improvement

📦 Business Impact

This system helps retailers:

Reduce overstock costs

Prevent stockouts

Improve reorder planning

Make data-driven inventory decisions
