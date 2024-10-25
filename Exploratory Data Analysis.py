# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays

# Load the datasets
sales_file_path = r"D:\Guvi\Project Dominos\Processed_Sales_Data.csv"
ingredients_file_path = r"D:\Guvi\Project Dominos\Processed_Ingredients.csv"
sales_df = pd.read_csv(sales_file_path)
ingredients_df = pd.read_csv(ingredients_file_path)

# Ensure 'order_date' is in datetime format
sales_df['order_date'] = pd.to_datetime(sales_df['order_date'])

# Extract the week number from 'order_date'
sales_df['week'] = sales_df['order_date'].dt.isocalendar().week

# Step 1: Filter the sales_df dataset for any week (week 7)
week_7_sales = sales_df[sales_df['week'] == 7]

# Step 2: Group by pizza name and sum the quantities
grouped_sales = week_7_sales.groupby('pizza_name')['quantity'].sum().reset_index()

# Step 3: List all grouped pizza names and their total quantities
print("Pizzas sold in week 7:")
for _, row in grouped_sales.iterrows():
    print(f"{row['pizza_name']}: {row['quantity']} pizzas")

# Step 4: Calculate and print the total quantity sold in week 7
total_quantity_sold = week_7_sales['quantity'].sum()
print(f"\nTotal units of pizza sold in week 7: {total_quantity_sold} pizzas")

# Extract additional time-based features
sales_df['month'] = sales_df['order_date'].dt.month
sales_df['day_of_week'] = sales_df['order_date'].dt.day_name()
sales_df['year'] = sales_df['order_date'].dt.year

# 1. Sales Trend Over Time
sales_trend = sales_df.groupby('order_date')['quantity'].sum().reset_index()

plt.figure(figsize=(15, 8))
plt.plot(sales_trend['order_date'], sales_trend['quantity'], marker='o', color='b')
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# 2. Monthly Sales Trends
sales_df['total_price'] = pd.to_numeric(sales_df['total_price'], errors='coerce')
monthly_sales = sales_df.resample('M', on='order_date')['total_price'].sum()

plt.figure(figsize=(12, 6))
monthly_sales.plot(color='orange')
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# 3. Sales by Day of the Week
weekly_sales = sales_df.groupby('day_of_week')['total_price'].sum()
ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_sales = weekly_sales.reindex(ordered_days)

plt.figure(figsize=(12, 6))
weekly_sales.plot(kind='bar', color='green')
plt.title('Total Sales by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# 4. Average Sales by Month (Bar Chart)
monthly_avg_sales = sales_df.groupby('month')['quantity'].mean()
plt.figure(figsize=(12, 6))
monthly_avg_sales.plot(kind='bar', color='skyblue')
plt.title('Average Sales by Month')
plt.xlabel('Month')
plt.ylabel('Average Quantity Sold')
plt.xticks(ticks=np.arange(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# 5. Monthly Quantity Sold Distribution (Bar Chart)
monthly_quantity = sales_df.groupby('month')['quantity'].sum()
plt.figure(figsize=(12, 6))
monthly_quantity.plot(kind='bar', color='lightgreen')
plt.title('Monthly Quantity Sold Distribution')
plt.xlabel('Month')
plt.ylabel('Total Quantity Sold')
plt.xticks(ticks=np.arange(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# 6. Total Sales by Pizza Category
category_sales = sales_df.groupby('pizza_category')['total_price'].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
category_sales.plot(kind='bar', color='purple')
plt.title('Total Sales by Pizza Category')
plt.xlabel('Pizza Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# 7. Total Sales by Pizza Size
size_sales = sales_df.groupby('pizza_size')['total_price'].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
size_sales.plot(kind='bar', color='cyan')
plt.title('Total Sales by Pizza Size')
plt.xlabel('Pizza Size')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# 8. Distribution of Unit Prices
plt.figure(figsize=(10, 6))
sns.histplot(sales_df['unit_price'], bins=30, kde=True, color='blue')
plt.title('Distribution of Unit Prices')
plt.xlabel('Unit Price')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()

# 9. Correlation Heatmap
numeric_df = sales_df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
