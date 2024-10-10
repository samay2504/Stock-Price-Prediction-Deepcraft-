# Stock Price Prediction

This project involves predicting stock prices using two models: LSTM and XGBoost. The goal is to explore time-series forecasting techniques and compare the performance of different approaches.

## Project Overview

This project aims to predict stock prices by utilizing machine learning models with sequential data. The key steps involved in the project are:

1. **Data Understanding and Exploratory Data Analysis (EDA)**
2. **Data Preprocessing and Feature Engineering**
3. **Model Selection and Training** (Using PyTorch LSTM)
4. **Model Evaluation and Result Analysis**
5. **Consideration of Improvements and Model Retraining** (Using XGBoost)

## Models Used

### LSTM (Long Short-Term Memory)
- Implemented using PyTorch.
- Trained over 200 epochs to achieve accurate results.
- Requires significant computational resources and time.

### XGBoost
- A more efficient model that provides accurate results with 100 boosting rounds.
- Works better for shorter time-series prediction without requiring high computational power.

## Data

The data used for this project is historical stock price data. Key features include:
- `Opening Price`
- `High Price`
- `Low Price`
- `Closing Price` (target variable)

Data file:
- `stock_price.csv`: Contains the historical price data for training and evaluation.

## Installation and Requirements

To run the project, install the necessary Python packages:
pip install -r requirements.txt

### Exploratory Data Analysis (EDA): 
Use the *eda.py* script for data visualization.

### Data Preparation: 
Run the *data_prep.py* script to preprocess the stock data.

### Model Training:
- For LSTM, run the *model.py*.
- For XGBoost, run the *improved.py*.

### Model Evaluation: 
Evaluate the model performance using the *evalve.py* script.

