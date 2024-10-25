# Import necessary libraries
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# Load the dataset
sales_df = pd.read_csv('D:\\Guvi\\Project Dominos\\final_sales_data.csv')

# Convert the order_date column to datetime format
sales_df['order_date'] = pd.to_datetime(sales_df['order_date'], errors='coerce')

# Group by date and pizza name, summing the quantity sold
daily_sales = sales_df.groupby(['order_date', 'pizza_name'])['quantity'].sum().reset_index()

# Create an empty DataFrame to store forecasts for each pizza
forecast_results = []

# Iterate through each unique pizza
for pizza in daily_sales['pizza_name'].unique():
    pizza_data = daily_sales[daily_sales['pizza_name'] == pizza]
    
    # Set the order_date as the index
    pizza_data.set_index('order_date', inplace=True)

    # Fit the ARIMA model (using order=(5, 1, 0))
    arima_model = sm.tsa.ARIMA(pizza_data['quantity'], order=(5, 1, 0))
    arima_result = arima_model.fit()

    # Sales Forecasting for the next week using the ARIMA model
    last_date = pizza_data.index.max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=7)

    # Forecast for the next week (7 days)
    forecast = arima_result.get_forecast(steps=7)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Create a DataFrame for the forecast results for the current pizza
    forecast_df = pd.DataFrame({
        'order_date': future_dates,
        'pizza_name': pizza,
        'predicted_quantity': forecast_mean.values,
        'lower_ci': forecast_ci.iloc[:, 0].values,
        'upper_ci': forecast_ci.iloc[:, 1].values
    })

    # Append forecast results for this pizza to the overall results
    forecast_results.append(forecast_df)

# Combine all forecasts into a single DataFrame
all_forecasts_df = pd.concat(forecast_results)

# Print forecast results for each pizza
print("Forecast Results:")
for index, row in all_forecasts_df.iterrows():
    print(f"Date: {row['order_date'].date()}, Pizza: {row['pizza_name']}, Predicted Quantity: {row['predicted_quantity']:.2f}, "
          f"Confidence Interval: ({row['lower_ci']:.2f}, {row['upper_ci']:.2f})")

# Pie Chart for Forecasted Quantities
# Aggregate forecast quantities for the next week for each pizza
total_forecast = all_forecasts_df.groupby('pizza_name')['predicted_quantity'].sum().reset_index()

# Plotting the pie chart
plt.figure(figsize=(10, 8))
plt.pie(total_forecast['predicted_quantity'], 
        labels=total_forecast['pizza_name'], 
        autopct='%1.1f%%', 
        startangle=140,
        colors=plt.cm.tab20.colors)  # Use a colormap for better color diversity

plt.title('Predicted Sales Distribution for Each Pizza for Next Week')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
