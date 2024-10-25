#Data cleaning ingredients

import pandas as pd

df = pd.read_csv(r"D:\Guvi\Project Dominos\Pizza ingredients.csv")

# Fill the null values for 'Items_Qty_In_Grams' based on conditions
df.loc[(df['pizza_ingredients'] == 'Caramelized Onions') & (df['Items_Qty_In_Grams'].isnull()), 'Items_Qty_In_Grams'] = 20.0
df.loc[(df['pizza_name_id'] == 'hawaiian_l') & (df['Items_Qty_In_Grams'].isnull()), 'Items_Qty_In_Grams'] = 60.0
df.loc[(df['pizza_name_id'] == 'hawaiian_m') & (df['Items_Qty_In_Grams'].isnull()), 'Items_Qty_In_Grams'] = 40.0
df.loc[(df['pizza_name_id'] == 'hawaiian_s') & (df['Items_Qty_In_Grams'].isnull()), 'Items_Qty_In_Grams'] = 20.0

# Convert 'Items_Qty_In_Grams' to integer type
df['Items_Qty_In_Grams'] = df['Items_Qty_In_Grams'].astype(int)

# Replace specific ingredient names for consistency
df['pizza_ingredients'] = df['pizza_ingredients'].replace({'Barbecued Chicken': "Barbecue Chicken", '?duja Salami': "nduja Salami"})

# Display basic information about the loaded ingredients data
df.info()

# Save the cleaned DataFrame to a new CSV file
Output_path = r"D:\Guvi\Project Dominos\Processed_Ingredients.csv"
df.to_csv(Output_path, index=False)
print("Processed ingredients data has been saved to:", Output_path)
