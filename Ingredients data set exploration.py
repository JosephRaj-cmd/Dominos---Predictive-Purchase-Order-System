import pandas as pd

# Load the dataset
df = pd.read_csv(r"D:\Guvi\Project Dominos\Pizza ingredients.csv")

# Display basic info about the DataFrame
print("DataFrame Info:")
df.info()

# Check for duplicate rows
duplicates = df.duplicated().sum()
print("Number of duplicate rows: ", duplicates)

# Remove duplicate rows if any
df = df.drop_duplicates()

# Verify that there are no more duplicates
duplicates_after = df.duplicated().sum()
print("Number of duplicate rows after cleaning: ", duplicates_after)

# Get unique values for each column
unique_values = {column: df[column].unique() for column in df.columns}

# Print the unique values
for column, values in unique_values.items():
    print(f"Unique values for column '{column}':")
    print(values)
    print()  # Print an empty line for better readability