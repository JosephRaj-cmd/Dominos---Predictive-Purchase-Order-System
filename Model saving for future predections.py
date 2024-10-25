import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error as mape
import warnings
import pickle

# Suppress warnings
warnings.filterwarnings("ignore")

# Step 1: Load the data
sales_df = pd.read_csv('D:\\Guvi\\Project Dominos\\final_sales_data.csv')
pizza_sales = sales_df  # Use the already loaded DataFrame


# Data Preparation - Aggregate pizza sales by week and pizza type
def prepare_weekly_sales(df):
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['week'] = df['order_date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_sales = df.groupby(['week', 'pizza_name_id'])['quantity'].sum().reset_index()
    weekly_sales.rename(columns={'week': 'ds', 'quantity': 'y'}, inplace=True)  # Renaming for consistency
    return weekly_sales

pizza_sales_weekly = prepare_weekly_sales(pizza_sales)

# Step 2: Train individual ARIMA models for each pizza_name_id
def train_arima_models_for_all_pizzas(pizza_sales_weekly):
    pizza_models = {}
    pizza_forecasts = {}
    pizza_mape_scores = []

    for pizza_id in pizza_sales_weekly['pizza_name_id'].unique():
        # Filter the sales data for this specific pizza
        pizza_data = pizza_sales_weekly[pizza_sales_weekly['pizza_name_id'] == pizza_id]
        pizza_data.set_index('ds', inplace=True)

        # Split the data into training and testing sets
        train_size = int(0.8 * len(pizza_data))
        train, test = pizza_data.iloc[:train_size], pizza_data.iloc[train_size:]

        # Train ARIMA model
        arima_model = ARIMA(train['y'], order=(1, 1, 1))  # Adjust ARIMA parameters based on best model analysis
        model_fit = arima_model.fit()

        # Forecast for the next week
        forecast = model_fit.get_forecast(steps=len(test) + 7)  # Including extra 7 days for next week's prediction
        forecast_values = forecast.predicted_mean[-7:]  # Extract the last 7 values as the next week's forecast

        # Calculate MAPE score for model performance
        predictions = forecast.predicted_mean[:len(test)]
        arima_mape = mape(test['y'], predictions)
        pizza_mape_scores.append({'pizza_name_id': pizza_id, 'mape': arima_mape})
        pizza_models[pizza_id] = model_fit

        # Save forecasted values for the next week
        pizza_forecasts[pizza_id] = pd.DataFrame({
            'ds': pd.date_range(start=pizza_data.index[-1] + pd.Timedelta(days=1), periods=7, freq='W'),
            'yhat': forecast_values.values
        })

    # Combine all forecasts into a single DataFrame
    all_forecasts = pd.concat([
        pd.DataFrame({
            'pizza_name_id': pizza_id, 
            'ds': pizza_forecasts[pizza_id]['ds'], 
            'yhat': pizza_forecasts[pizza_id]['yhat']
        })
        for pizza_id in pizza_forecasts
    ], ignore_index=True)

    return pizza_models, pizza_mape_scores, all_forecasts

# Step 3: Train the models
pizza_models, pizza_mape_scores, predicted_sales_weekly = train_arima_models_for_all_pizzas(pizza_sales_weekly)

# Step 4: Load the ingredient dataset and calculate ingredient requirements
ingredients_df = pd.DataFrame({
    'pizza_name_id': ['A', 'B', 'C'],  # Example data; replace with actual ingredient data
    'pizza_ingredients': ['Cheese', 'Tomato', 'Pepperoni'],
    'Items_Qty_In_Grams': [150, 100, 200]  # Amount of each ingredient per pizza
})

# Merge the predicted weekly sales with the ingredient dataset
ingredient_requirements = predicted_sales_weekly.merge(ingredients_df, on='pizza_name_id', how='left')

# Calculate the total required quantity of each ingredient for the predicted weekly sales
ingredient_requirements['total_ingredient_qty'] = ingredient_requirements['yhat'] * ingredient_requirements['Items_Qty_In_Grams']

# Group by ingredient to get the total quantity needed for each over the week
weekly_ingredient_totals = ingredient_requirements.groupby('pizza_ingredients')['total_ingredient_qty'].sum().reset_index()

# Step 5: Save trained models to a file
with open(r"D:\\Guvi\\Project Dominos\\trained_model.pkl", 'wb') as f:
    pickle.dump(pizza_models, f)

# Print MAPE for each pizza and the weekly ingredient requirements
print("Weekly Ingredient Requirements:")
print(weekly_ingredient_totals)

print("\nModel Performance (MAPE):")
for item in pizza_mape_scores:
    print(f'Pizza {item["pizza_name_id"]}: MAPE = {item["mape"]:.2f}')
