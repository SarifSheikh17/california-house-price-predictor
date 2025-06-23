# California House Price Predictor (ML + Streamlit)

This project uses regression models to predict California housing prices using features from the [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). It includes data preprocessing, training, evaluation, and an interactive Streamlit application.

---

## Features Used

The following input features were used to predict the median house value:

- `MedInc` (Median income in block)
- `HouseAge` (Average age of houses in the block)
- `AveRooms` (Average number of rooms per household)
- `AveBedrms` (Average number of bedrooms per household)
- `Population` (Block population)
- `AveOccup` (Average occupancy)
- `Latitude` (Location coordinate)
- `Longitude` (Location coordinate)

---

## Models Trained

- Linear Regression
- MLP Regressor (Multi-layer Perceptron Neural Network)

---

## Evaluation Metrics

| Model            | MAE     | MSE      | RMSE     | R² Score |
|------------------|---------|----------|----------|----------|
| Linear Regression| ~       | ~        | ~        | ~        |
| MLP Regressor    | ~       | ~        | ~        | ~        |

*Replace the `~` with actual results after training.*

---

## Streamlit App

### Features:

- Responsive slider inputs for all 8 features
- Predictions from both models displayed side-by-side
- Clean layout with sidebar information
- Real-time results with model confidence

---

## Getting Started

### Step 1: Clone the Repository

```bash
git clone https://github.com/SarifSheikh17/california-house-price-predictor.git
cd california-house-price-predictor
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train the Models

```bash
python main.py
```

This will save the following files in the `models/` directory:

- `linear_regression_model.pkl`
- `mlp_regressor_model.pkl`
- `scaler.pkl`
- `selected_features.pkl`

### Step 4: Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## Project Structure

```bash
california-house-price-predictor/
├── main.py
├── streamlit_app.py
├── models/
│   ├── linear_regression_model.pkl
│   ├── mlp_regressor_model.pkl
│   ├── scaler.pkl
│   └── selected_features.pkl
├── requirements.txt
├── README.md
```

---

## Acknowledgements

Dataset: `fetch_california_housing` from `sklearn.datasets`  
Libraries: Scikit-learn, Matplotlib, Seaborn, Streamlit  
Developed through hands-on practice with help from ChatGPT

---

## Author

Sarif Sheikh  
ML Intern at IIITDM Kancheepuram
