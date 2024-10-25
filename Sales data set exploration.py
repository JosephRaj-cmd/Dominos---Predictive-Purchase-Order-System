#Dataset Exploration - Sales

import pandas as pd

# Load your dataset
df = pd.read_csv(r"D:\Guvi\Project Dominos\Pizza sales.csv")

# Check the DataFrame type
print(type(df))  

df.head() 

print("Dataset Overview:")
print(df.info())  # Data types, non-null count
print("\nSummary Statistics for Numerical Columns:")
print(df.describe())  # Summary statistics for numerical columns


# Count missing values per column
print("\nMissing Data Count per Column:")
missing_values = df.isnull().sum()
print(missing_values)

# Display rows that have missing data
print("\nRows with Missing Data:")
print(df[df.isnull().any(axis=1)])  # Display rows with missing values

# Unique Value Counts per Column

columns_of_interest = [
    'pizza_id', 'order_id', 'pizza_name_id', 'quantity', 'order_date', 
    'order_time', 'unit_price', 'total_price', 'pizza_size', 
    'pizza_category', 'pizza_ingredients'
]

# Print the number of unique values in each column
print("\nUnique Values Count per Column:")
for col in columns_of_interest:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count} unique values")

# Display the unique values for each column
print("\nUnique Values for Each Column:")
for col in columns_of_interest:
    print(f"\nUnique values in {col}:")
    print(df[col].unique())

# Check for duplicate rows
duplicate_rows = df[df.duplicated()]
print(f"\nNumber of Duplicate Rows: {len(duplicate_rows)}")
print(duplicate_rows)

# Display rows with missing values in the pizza_ingredients column
missing_ingredients = df.loc[df['pizza_ingredients'].isnull()]

# Print the rows with missing values
print("Rows with missing values in 'pizza_ingredients':")
print(missing_ingredients)