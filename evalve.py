import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('stock_price.csv')

# Convert 'Date' column to datetime for better time-series handling
df['Date'] = pd.to_datetime(df['Date'])

# Handle any missing values using forward fill
df.ffill(inplace=True)

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
train_size = int(len(X_tensor) * 0.7)  # 70% for training
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)  # Output layer (predicting single value)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size)  # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size)  # cell state

        # Propagate input through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        # Take the last time step's output
        out = lstm_out[:, -1, :]  # Take the output of the last time step

        # Pass the last time step's output through the fully connected layer
        predictions = self.fc(out)
        return predictions


# Hyperparameters
input_size = 3  # Number of features
hidden_size = 50  # Number of LSTM units
num_layers = 1  # Number of LSTM layers
num_epochs = 200  # Number of training epochs
learning_rate = 0.001  # Learning rate

# Initialize the model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear gradients

    # Forward pass
    output = model(X_train)
    loss = criterion(output, y_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing the model and making predictions on the test set
model.eval()  # Set the model to evaluation mode
test_predictions = []

for seq in X_test:
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        test_predictions.append(model(seq.unsqueeze(0)).item())

# Inverse the scaling for both predicted and true prices
test_predictions = scaler_y.inverse_transform(np.array(test_predictions).reshape(-1, 1))
y_test_actual = scaler_y.inverse_transform(y_test.detach().numpy().reshape(-1, 1))

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test_actual, test_predictions)
print(f"Mean Squared Error: {mse}")

# Plot the actual vs predicted stock prices
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, label='True Prices')
plt.plot(test_predictions, label='Predicted Prices')
plt.title("LSTM Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
