# Import core libraries
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Scikit-learn dataset loader
from sklearn.datasets import fetch_california_housing

# Load the dataset
california = fetch_california_housing()

# Convert to DataFrame
data = pd.DataFrame(california.data, columns=california.feature_names)
data['MedHouseValue'] = california.target  # Target column

# Show the first 5 rows
print(data.head())

# Basic dataset info
print("Shape of dataset:", data.shape)
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

import plotly.express as px

fig = px.histogram(data, x='MedHouseValue', nbins=50, title="Distribution of House Prices")
#fig.show()

# Find top 2 correlated features with target
correlations = data.corr()['MedHouseValue'].drop('MedHouseValue')
top_features = correlations.abs().sort_values(ascending=False).head(2).index.tolist()

# Plot scatter plots with trendline
#for feature in top_features:
 #   fig = px.scatter(data, x=feature, y='MedHouseValue', trendline="ols",
  #                   title=f"{feature} vs House Price")
   # fig.show()

import seaborn as sns

# Use only a few columns to avoid overloading
subset = data[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'MedHouseValue']]
sns.pairplot(subset)
plt.show()

X = data.drop('MedHouseValue', axis=1)
y = data['MedHouseValue']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression

# Initialize model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train_scaled, y_train)

y_pred = lr_model.predict(X_test_scaled)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted Prices (Linear Regression)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid(True)
plt.show()


import joblib
joblib.dump(lr_model, "linear_model.pkl")
joblib.dump(scaler, "scaler.pkl")


