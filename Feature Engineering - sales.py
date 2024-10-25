#Feature Engineering - sales

import pandas as pd
import matplotlib.pyplot as plt
import holidays

# Load the data
file_path = r"D:\Guvi\Project Dominos\Processed_Sales_Data.csv"
df = pd.read_csv(file_path)

# Assuming df has columns 'order_date' and 'quantity'
# Group by 'order_date' and sum 'quantity'
df_daily_count = df.groupby('order_date', as_index=False).agg({'quantity': 'sum'})

# Calculate the median sales threshold
Median_sales_threshold = df_daily_count['quantity'].median()

# Load US holidays for 2015
us_holidays = holidays.US(years=2015)

# Function to classify the date as 'Normal', 'Holiday', or 'Promotional Period'
def classify_sales(row):
    if row['quantity'] > Median_sales_threshold:
        if row['order_date'] in us_holidays:
            return 'Holiday'
        else:
            return 'Promotional Period'
    else:
        return 'Normal Sales'
    
    # Apply the classification function
df_daily_count['sales_type'] = df_daily_count.apply(classify_sales, axis=1)

# Check the classified DataFrame
print(df_daily_count.head())

# Optional: Visualize the classified sales types
df_daily_count['sales_type'].value_counts().plot(kind='bar', title='Sales Type Distribution')
plt.xlabel('Sales Type')
plt.ylabel('Count')
plt.show()

# Convert sales_type to dummy variables
df_daily_count = pd.get_dummies(df_daily_count, columns=['sales_type'])

# Rename columns for clarity
df_daily_count = df_daily_count.rename(columns={
    'sales_type_Holiday': 'Holiday',
    'sales_type_Normal Sales': 'Normal Sales',
    'sales_type_Promotional Period': 'Promotional Period'
})

# Convert boolean columns to integer
df_daily_count['Holiday'] = df_daily_count['Holiday'].map({True: 1, False: 0})
df_daily_count['Normal Sales'] = df_daily_count['Normal Sales'].map({True: 1, False: 0})
df_daily_count['Promotional Period'] = df_daily_count['Promotional Period'].map({True: 1, False: 0})

# Merge the daily count with the original DataFrame
df = pd.merge(df, df_daily_count, on='order_date', how='inner')

# Rename columns to avoid conflicts
df = df.rename(columns={'quantity_x': 'quantity', 'quantity_y': 'Total quantity per day'})

# Save the output DataFrame to a CSV file
output_file_path = r"D:\Guvi\Project Dominos\final_sales_data.csv"
df.to_csv(output_file_path, index=False)

print("Output saved to:", output_file_path)