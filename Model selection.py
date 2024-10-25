# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from prophet import Prophet

# Load the dataset
sales_df = pd.read_csv('D:\\Guvi\\Project Dominos\\final_sales_data.csv')

# Convert the order_date column to datetime format
sales_df['order_date'] = pd.to_datetime(sales_df['order_date'], errors='coerce')

# Extract additional time-based features
sales_df['month'] = sales_df['order_date'].dt.month
sales_df['year'] = sales_df['order_date'].dt.year
sales_df['day_of_year'] = sales_df['order_date'].dt.dayofyear
sales_df['week_of_year'] = sales_df['order_date'].dt.isocalendar().week

# One-hot encode the 'day_of_week' column
sales_df = pd.get_dummies(sales_df, columns=['day_of_week'], drop_first=True)

# Prepare the data for regression
X = sales_df[['day_of_year', 'week_of_year', 'month'] + [col for col in sales_df.columns if col.startswith('day_of_week_')]]
y = sales_df['quantity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 1. Linear Regression Model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
regression_predictions = regression_model.predict(X_test)
regression_mape = mean_absolute_percentage_error(y_test, regression_predictions)

# 2. ARIMA Model
arima_model = sm.tsa.ARIMA(y_train, order=(5, 1, 0))
arima_result = arima_model.fit()
arima_predictions = arima_result.forecast(steps=len(y_test))
arima_mape = mean_absolute_percentage_error(y_test, arima_predictions)

# 3. SARIMA Model
sarima_model = sm.tsa.SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()
sarima_predictions = sarima_result.forecast(steps=len(y_test))
sarima_mape = mean_absolute_percentage_error(y_test, sarima_predictions)

# 4. Decision Tree Regressor
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_mape = mean_absolute_percentage_error(y_test, dt_predictions)

# 5. Random Forest Regressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mape = mean_absolute_percentage_error(y_test, rf_predictions)

# 6. Gradient Boosting Regressor
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
gb_mape = mean_absolute_percentage_error(y_test, gb_predictions)

# 7. LSTM Model
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

try:
    # Scale the features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    # Scale the target variable
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

    # Create training and testing datasets
    time_step = 10
    X_train_lstm, y_train_lstm = create_dataset(y_train_scaled, time_step)
    X_test_lstm, y_test_lstm = create_dataset(y_test_scaled, time_step)

    # Reshape the data for LSTM [samples, time steps, features]
    X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], 1)
    X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], 1)

    # Build the LSTM model
    lstm_model = Sequential([
        Input(shape=(time_step, 1)),
        LSTM(25, return_sequences=False),
        Dense(1)
    ])

    # Compile the model
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model with Early Stopping
    early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
    lstm_model.fit(X_train_lstm, y_train_lstm, batch_size=32, epochs=5, callbacks=[early_stopping])

    # Make predictions
    train_predict = lstm_model.predict(X_train_lstm)
    test_predict = lstm_model.predict(X_test_lstm)

    # Inverse transform to get actual values
    test_predict = target_scaler.inverse_transform(test_predict)
    y_test_inverse = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1))

    # Calculate LSTM metrics
    lstm_mape = mean_absolute_percentage_error(y_test_inverse, test_predict)
except Exception as e:
    print(f"LSTM Error: {e}")
    lstm_mape = None  # Set to None if there's an error

# 8. Prophet Model
try:
    # Prepare data for Prophet
    prophet_df = sales_df[['order_date', 'quantity']].rename(columns={'order_date': 'ds', 'quantity': 'y'})
    prophet_model = Prophet()
    prophet_model.fit(prophet_df)

    # Create future dataframe
    future = prophet_model.make_future_dataframe(periods=len(y_test))
    forecast = prophet_model.predict(future)

    # Calculate Prophet metrics
    prophet_predictions = forecast['yhat'][-len(y_test):]
    prophet_mape = mean_absolute_percentage_error(y_test, prophet_predictions)
except Exception as e:
    print(f"Prophet Error: {e}")
    prophet_mape = None  # Set to None if there's an error

# Choose the best model based on MAPE
mape_results = [
    (regression_mape, 'Linear Regression'),
    (arima_mape, 'ARIMA'),
    (sarima_mape, 'SARIMA'),
    (lstm_mape, 'LSTM') if lstm_mape is not None else (None, 'LSTM'),
    (dt_mape, 'Decision Tree'),
    (rf_mape, 'Random Forest'),
    (gb_mape, 'Gradient Boosting'),
    (prophet_mape, 'Prophet') if prophet_mape is not None else (None, 'Prophet'),
]

# Filter out None values
mape_results = [result for result in mape_results if result[0] is not None]

# Identify the best model
best_model = min(mape_results, key=lambda x: x[0])
print(f"Best Model: {best_model[1]} with MAPE: {best_model[0]}")

# Print all MAPE results
print("MAPE Results:")
for mape, model_name in mape_results:
    print(f"{model_name}: {mape}")
