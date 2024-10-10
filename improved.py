import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('D:/Projects/Deepcraft/assignment-main/Trainee/time-series-prediction/stock_price.csv')

# Convert 'Date' column to datetime for better time-series handling
df['Date'] = pd.to_datetime(df['Date'])

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Feature Engineering: Creating lag features
def create_lagged_features(df, lag=5):
    """
    Create lagged features based on previous n days (lag).
    """
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df['Closing Price'].shift(i)
    return df

# Create lagged features
df = create_lagged_features(df, lag=5)

# Drop rows with missing values (due to lagging)
df.dropna(inplace=True)

# Features and Target selection
X = df[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']]  # Lag features as input
y = df['Closing Price']  # Closing Price as the target

# Normalize the target variable (optional)
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler_y.fit_transform(np.array(y).reshape(-1, 1))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.3, shuffle=False)

# Convert data into DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# XGBoost parameters
params = {
    'objective': 'reg:squarederror',  # For regression tasks
    'eval_metric': 'rmse',
    'max_depth': 3,                   # Maximum depth of the trees
    'eta': 0.1,                       # Learning rate
    'subsample': 0.8,                 # Subsampling ratio
    'colsample_bytree': 0.8,          # Ratio of columns used per tree
}

# Train the model
num_boost_round = 100
model_xgb = xgb.train(params, dtrain, num_boost_round)

# Predict on the test set
y_pred = model_xgb.predict(dtest)

# Inverse the scaling for the predicted prices
y_pred_inverse = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
y_test_inverse = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Calculate Mean Squared Error (MSE)
mse_xgb = mean_squared_error(y_test_inverse, y_pred_inverse)
print(f'Mean Squared Error (XGBoost): {mse_xgb}')

# Plot the actual vs predicted stock prices
plt.figure(figsize=(10, 6))
plt.plot(df['Date'].iloc[-len(y_test):], y_test_inverse, label='True Prices')
plt.plot(df['Date'].iloc[-len(y_test):], y_pred_inverse, label='XGBoost Predictions', color='orange')
plt.title('XGBoost Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
