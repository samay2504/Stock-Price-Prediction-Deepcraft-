import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('stock_price.csv')

df['Date'] = pd.to_datetime(df['Date'])

df.ffill(inplace=True)

X = df[['Opening Price', 'High Price', 'Low Price']]
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

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output layer

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last output
        return out

# Hyperparameters
input_size = 3  # Number of features
hidden_size = 50  # Number of LSTM units
num_layers = 1  # Number of LSTM layers
num_epochs = 100  # Number of training epochs
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

# Testing the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_pred = model(X_test)

# Inverse transform to get actual closing prices
y_pred_inverse = scaler_y.inverse_transform(y_pred.numpy())
y_test_inverse = scaler_y.inverse_transform(y_test.numpy().reshape(-1, 1))

# Binarizing predictions for confusion matrix (optional)
threshold = 0.5  # Set a threshold for binary classification
y_pred_class = (y_pred_inverse > threshold).astype(int)
y_test_class = (y_test_inverse > threshold).astype(int)

# Confusion matrix and accuracy
cm = confusion_matrix(y_test_class, y_pred_class)
accuracy = accuracy_score(y_test_class, y_pred_class)

# Print the confusion matrix and accuracy
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Optional: Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
