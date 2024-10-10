import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv('stock_price.csv')

# Convert 'Date' column to datetime for better time-series handling
df['Date'] = pd.to_datetime(df['Date'])

# Display first few rows and data types
print("First 5 rows of the dataset:")
print(df.head())

print("\nData types and missing values:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Set the default plot style for seaborn to enhance aesthetics
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

plt.figure(figsize=(14, 10))
for i, col in enumerate(['Closing Price', 'Opening Price', 'High Price', 'Low Price'], 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[col], bins=50, color='#87CEEB', kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()

plt.show()

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 16), sharex=True)

# Color palette for the lines
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red

# List of columns to plot
cols = ['Closing Price', 'Opening Price', 'High Price', 'Low Price']

axes = axes.ravel()

for i, (col, color) in enumerate(zip(cols, colors)):
    axes[i].plot(df['Date'], df[col], color=color, linewidth=2, alpha=0.8)
    axes[i].set_title(f"{col} Over Time", fontsize=14)
    axes[i].set_ylabel('Price')
    axes[i].grid(True)

# Rotate x-ticks for better readability
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.tight_layout()
plt.show()

# Enhanced correlation heatmap with more legible colors
plt.figure(figsize=(12, 8))
corr_matrix = df[['Closing Price', 'Opening Price', 'High Price', 'Low Price']].corr()

# Use a diverging colormap and reduce annotation size for clarity
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='black', annot_kws={"size": 12})
plt.title("Correlation Matrix of Stock Prices")
plt.tight_layout()
plt.show()
