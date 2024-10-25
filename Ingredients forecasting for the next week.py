# Import necessary libraries
import pandas as pd
import pickle
import warnings
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# Load trained models
with open(r"D:\Guvi\Project Dominos\trained_model.pkl", 'rb') as f:
    pizza_models = pickle.load(f)

# Load the pizza sales data
sales_df = pd.read_csv('D:\\Guvi\\Project Dominos\\final_sales_data.csv')

# Load the ingredients dataset
ingredients_df = pd.read_csv(r"D:\Guvi\Project Dominos\Processed_Ingredients.csv")

# Prepare weekly sales data by pizza type
def prepare_weekly_sales_by_pizza(df):
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['week'] = df['order_date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_sales_by_pizza = df.groupby(['week', 'pizza_name_id'])['quantity'].sum().reset_index()
    weekly_sales_by_pizza.rename(columns={'week': 'ds', 'quantity': 'y'}, inplace=True)
    return weekly_sales_by_pizza

pizza_sales_weekly_by_pizza = prepare_weekly_sales_by_pizza(sales_df)

# Predict next week's sales for each pizza type using ARIMA
def forecast_next_week_sales(pizza_models, pizza_sales_weekly_by_pizza, periods=7):
    pizza_forecasts = {}
    
    for pizza_id in pizza_sales_weekly_by_pizza['pizza_name_id'].unique():
        pizza_data = pizza_sales_weekly_by_pizza[pizza_sales_weekly_by_pizza['pizza_name_id'] == pizza_id]
        pizza_data.set_index('ds', inplace=True)
        
        # Retrieve the trained ARIMA model for this pizza_id
        model = pizza_models[pizza_id]

        # Forecast for the next 7 days (one week)
        forecast = model.get_forecast(steps=periods)
        forecast_values = forecast.predicted_mean  # Get the forecasted values for the next week

        # Store the forecasted quantity
        pizza_forecasts[pizza_id] = forecast_values
    
    return pizza_forecasts

# Calculate the ingredient purchase order
def calculate_purchase_order(pizza_forecasts):
    # Copy ingredients and map predicted sales
    ingredients = ingredients_df.copy()
    ingredients['predicted_quantity'] = ingredients['pizza_name_id'].map({k: v.sum() for k, v in pizza_forecasts.items()})

    # Calculate total ingredient quantity needed in grams
    ingredients['total_ingredient_qty_grams'] = ingredients['Items_Qty_In_Grams'] * ingredients['predicted_quantity']
    
    # Calculate daily, weekly, monthly, and yearly requirements in grams
    ingredients['daily_qty'] = ingredients['predicted_quantity'] / 7  # Daily Requirement
    ingredients['weekly_qty'] = ingredients['predicted_quantity']      # Weekly Requirement
    ingredients['monthly_qty'] = ingredients['predicted_quantity'] * 4  # Monthly Requirement
    ingredients['yearly_qty'] = ingredients['predicted_quantity'] * 52  # Yearly Requirement

    return ingredients

# Step 1: Generate the forecasts
next_week_pizza_sales_forecasts = forecast_next_week_sales(pizza_models, pizza_sales_weekly_by_pizza, periods=7)

# Step 2: Generate the purchase order
ingredient_requirements = calculate_purchase_order(next_week_pizza_sales_forecasts)

# Step 3: Print daily, weekly, monthly, and yearly sums of ingredients in grams
print("\nDaily Ingredient Requirements (in grams):")
print(ingredient_requirements[['pizza_ingredients', 'daily_qty']])

print("\nWeekly Ingredient Requirements (in grams):")
print(ingredient_requirements[['pizza_ingredients', 'weekly_qty']])

print("\nMonthly Ingredient Requirements (in grams):")
print(ingredient_requirements[['pizza_ingredients', 'monthly_qty']])

print("\nYearly Ingredient Requirements (in grams):")
print(ingredient_requirements[['pizza_ingredients', 'yearly_qty']])

# Optional: Visualize the results as bar plots
def plot_ingredient_requirements(ingredients):
    # Daily requirements
    plt.figure(figsize=(10, 5))
    plt.bar(ingredients['pizza_ingredients'], ingredients['daily_qty'], color='skyblue')
    plt.title('Daily Ingredient Requirements (grams)')
    plt.xlabel('Pizza Ingredients')
    plt.ylabel('Daily Quantity (grams)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Weekly requirements
    plt.figure(figsize=(10, 5))
    plt.bar(ingredients['pizza_ingredients'], ingredients['weekly_qty'], color='lightgreen')
    plt.title('Weekly Ingredient Requirements (grams)')
    plt.xlabel('Pizza Ingredients')
    plt.ylabel('Weekly Quantity (grams)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Monthly requirements
    plt.figure(figsize=(10, 5))
    plt.bar(ingredients['pizza_ingredients'], ingredients['monthly_qty'], color='gold')
    plt.title('Monthly Ingredient Requirements (grams)')
    plt.xlabel('Pizza Ingredients')
    plt.ylabel('Monthly Quantity (grams)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Yearly requirements
    plt.figure(figsize=(10, 5))
    plt.bar(ingredients['pizza_ingredients'], ingredients['yearly_qty'], color='salmon')
    plt.title('Yearly Ingredient Requirements (grams)')
    plt.xlabel('Pizza Ingredients')
    plt.ylabel('Yearly Quantity (grams)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot ingredient requirements
plot_ingredient_requirements(ingredient_requirements)
