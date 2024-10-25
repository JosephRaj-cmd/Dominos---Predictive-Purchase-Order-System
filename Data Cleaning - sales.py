#Data Cleaning - sales

import pandas as pd
import matplotlib.pyplot as plt
import holidays

# Load your dataset
df = pd.read_csv(r"D:\Guvi\Project Dominos\Pizza sales.csv")

# Step 1: Fill missing values
print("Missing values before filling:\n", df.isnull().sum())

# List of columns with missing values
columns_with_missing_values = ['pizza_name_id', 'total_price', 'pizza_category', 'pizza_ingredients', 'pizza_name']

# Function to fill missing values in the specified column using other columns
def fill_missing_values(data, column_to_fill, reference_columns):
    for idx, row in data[data[column_to_fill].isnull()].iterrows():
        ref_values = row[reference_columns]
        matching_row = data[(data[reference_columns] == ref_values).all(axis=1) & data[column_to_fill].notnull()]
        if not matching_row.empty:
            data.at[idx, column_to_fill] = matching_row[column_to_fill].values[0]

# Apply the function to each column with missing values
for column in columns_with_missing_values:
    ref_columns = [col for col in columns_with_missing_values if col != column]
    fill_missing_values(df, column, ref_columns)

print("Missing values after filling:\n", df.isnull().sum())


# Step 2: Remove outliers
def remove_outliers(df):
    # Get numerical columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    for numerical_column in numerical_columns:
        print(f"\nProcessing column: {numerical_column}")
        
        # Print rows before removing outliers
        print("Rows before removing outliers:")
        print(df[numerical_column].describe())
        
        # Calculate Q1, Q3, and IQR
        Q1 = df[numerical_column].quantile(0.25)
        Q3 = df[numerical_column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Filter out outliers
        df = df[(df[numerical_column] >= (Q1 - 1.5 * IQR)) & (df[numerical_column] <= (Q3 + 1.5 * IQR))]
        
        # Print rows after removing outliers
        print("\nRows after removing outliers:")
        print(df[numerical_column].describe())
    
    return df
df_filtered = remove_outliers(df)

# Step 3: Calculate statistics for numerical columns
stats = {
    'mean': [df_filtered['unit_price'].mean(), df_filtered['total_price'].mean(), df_filtered['quantity'].mean()],
    'median': [df_filtered['unit_price'].median(), df_filtered['total_price'].median(), df_filtered['quantity'].median()],
    'mode': [df_filtered['unit_price'].mode()[0], df_filtered['total_price'].mode()[0], df_filtered['quantity'].mode()[0]],
    'std': [df_filtered['unit_price'].std(), df_filtered['total_price'].std(), df_filtered['quantity'].std()],
    'var': [df_filtered['unit_price'].var(), df_filtered['total_price'].var(), df_filtered['quantity'].var()]
}

#4. View in DataFrame
compare = pd.DataFrame(stats, index=['unit_price', 'total_price', 'quantity'])
print(compare)

# Step 5: Parse 'order_date' column
# Define a list of date formats to try
date_formats = ['%d-%m-%Y', '%d/%m/%Y']

def parse_dates(date_str):
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    return pd.NaT

# Apply the parsing function to the 'order_date' column
df['order_date'] = df['order_date'].apply(parse_dates)

# Convert to the desired format
df['order_date'] = df['order_date'].dt.strftime('%d-%m-%Y')

# Convert the date column to datetime
df['order_date'] = pd.to_datetime(df['order_date'], format='%d-%m-%Y')

# Step 6: Extract time features
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month_name()
df['day_of_week'] = df['order_date'].dt.day_name()

# Verify the data types
print(df.dtypes)

# Step 7: Group by 'order_date' and sum 'quantity'
df_daily_count = df.groupby('order_date', as_index=False).agg({'quantity': 'sum'})

# Calculate mean and median sales thresholds
Mean_sales_threshold = df_daily_count['quantity'].mean()
Median_sales_threshold = df_daily_count['quantity'].median()

# Step 8: Visualize sales distribution with the mean
df_daily_count['quantity'].plot(kind='hist', bins=30, title='Sales Distribution with Mean')
plt.axvline(Mean_sales_threshold, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {Mean_sales_threshold:.2f}')
plt.legend()
plt.show()

# Step 9: Visualize sales distribution with the median
df_daily_count['quantity'].plot(kind='hist', bins=30, title='Sales Distribution with Median')
plt.axvline(Median_sales_threshold, color='r', linestyle='dashed', linewidth=2, label=f'Median: {Median_sales_threshold:.2f}')
plt.legend()
plt.show()

# Save the output as a CSV file
output_file_path = r"D:\Guvi\Project Dominos\Processed_Sales_Data.csv"
df.to_csv(output_file_path, index=False)

print("Processed data has been saved to:", output_file_path)