# 🏠 California House Price Predictor

A simple yet powerful machine learning web app that predicts house prices in California based on multiple input features like median income, house age, number of rooms, and geographical location.

Built using:
- **Scikit-learn** for model training
- **Streamlit** for the interactive web interface
- **California Housing Dataset** from scikit-learn

---

## 🚀 Demo
> [Demo Screenshot](demo.png)

---

## 📦 Features

- 🔢 Predict house prices based on 8 real-estate factors
- 📊 Interactive web interface using Streamlit
- 📁 Easy to run locally with minimal setup
- 📈 Trained using Linear Regression
- ✅ Scaled inputs for better model performance
- 💾 Model and Scaler saved with `joblib`

---

## 📁 Dataset

We use the **California Housing Dataset** available directly from `sklearn.datasets.fetch_california_housing()`.

This dataset contains:
- Median income
- House age
- Average rooms & bedrooms
- Population
- Average occupants
- Latitude & Longitude
- Target: Median house value (in $100,000s)

---

## 🛠️ Installation & Running Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/SarifSheikh17/california-house-price-predictor.git
   cd California-House-Price-Predictor
