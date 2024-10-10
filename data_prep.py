import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file
df = pd.read_csv('D:/Projects/Deepcraft/assignment-main/Trainee/time-series-prediction/stock_price.csv')

# Convert 'Date' column to datetime for better time-series handling
df['Date'] = pd.to_datetime(df['Date'])

# Handle any missing values using forward fill
df.ffill(inplace=True)  # Forward fill any missing values

# Features and Target selection
X = df[['Opening Price', 'High Price', 'Low Price']]  # Use other price columns as features
y = df['Closing Price']  # Target variable

# Normalize the features and target
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler_y.fit_transform(np.array(y).reshape(-1, 1))

# Convert to tensors for PyTorch
X_tensor = torch.from_numpy(X_scaled).float()
y_tensor = torch.from_numpy(y_scaled).float()

# Reshape X for LSTM input (batch_size, sequence_length, number_of_features)
X_tensor = X_tensor.unsqueeze(1)  # Add sequence length of 1

# Splitting the dataset into training and testing sets
train_size = int(len(X_tensor) * 0.8)  # 80% for training
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Print the shapes of the training and test data
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
