# Purchase order calculation using ARIMA

import pandas as pd
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load trained models
with open(r"D:\Guvi\Project Dominos\trained_model.pkl", 'rb') as f:
    pizza_models = pickle.load(f)

# Load the pizza sales data
sales_df = pd.read_csv('D:\\Guvi\\Project Dominos\\final_sales_data.csv')

# Step 1: Prepare weekly sales by pizza type
def prepare_weekly_sales_by_pizza(df):
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['week'] = df['order_date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_sales_by_pizza = df.groupby(['week', 'pizza_name_id'])['quantity'].sum().reset_index()
    weekly_sales_by_pizza.rename(columns={'week': 'ds', 'quantity': 'y'}, inplace=True)
    return weekly_sales_by_pizza

pizza_sales_weekly_by_pizza = prepare_weekly_sales_by_pizza(sales_df)

# Step 2: Predict next week's sales for each pizza type using ARIMA
def forecast_next_week_sales(pizza_models, pizza_sales_weekly_by_pizza, periods=7):
    pizza_forecasts = {}

    for pizza_id in pizza_sales_weekly_by_pizza['pizza_name_id'].unique():
        pizza_data = pizza_sales_weekly_by_pizza[pizza_sales_weekly_by_pizza['pizza_name_id'] == pizza_id]
        pizza_data.set_index('ds', inplace=True)
        
        # Retrieve the trained ARIMA model for this pizza_id
        model = pizza_models[pizza_id]

        # Forecast for the next 7 days (one week)
        forecast = model.get_forecast(steps=periods)
        forecast_values = forecast.predicted_mean.sum()  # Sum the forecasted values for the next week

        # Store the forecasted quantity
        pizza_forecasts[pizza_id] = forecast_values
    
    return pizza_forecasts

# Step 3: Calculate the ingredient purchase order
def calculate_purchase_order(pizza_forecasts):
    # Load the ingredient data
    ingredients = pd.read_csv(r"D:\Guvi\Project Dominos\Processed_Ingredients.csv")
    
    # Copy ingredients and map predicted sales
    ingredients['predicted_quantity'] = ingredients['pizza_name_id'].map(pizza_forecasts)
    
    # Calculate total ingredient quantity needed
    ingredients['total_ingredient_qty'] = ingredients['Items_Qty_In_Grams'] * ingredients['predicted_quantity']
    
    # Group by ingredient type and sum the total quantity
    ingredient_totals = ingredients.groupby('pizza_ingredients')['total_ingredient_qty'].sum().reset_index()
    ingredient_totals['total_ingredient_qty_kg'] = ingredient_totals['total_ingredient_qty'] / 1000  # Convert to kg
    
    return ingredient_totals

# Step 4: Generate the forecasts and purchase order
next_week_pizza_sales_forecasts = forecast_next_week_sales(pizza_models, pizza_sales_weekly_by_pizza, periods=7)
purchase_order = calculate_purchase_order(next_week_pizza_sales_forecasts)

# Step 5: Save the purchase order to CSV
purchase_order_output_path = r"D:\Guvi\Project Dominos\Purchase_Order.csv"
purchase_order.to_csv(purchase_order_output_path, index=False)

# Print the purchase order
print("Purchase Order for Next Week:")
print(purchase_order)
