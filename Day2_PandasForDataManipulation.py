import os

import pandas as pd
df = pd.read_csv("Day2_Data.csv")

print("Shape:", df.shape)          
print("\nData Types:\n", df.dtypes)  
print("\nFirst 10 rows Data:\n", df.head(10))

# Filter: Display only successful payments
df_filtered = df[df['payment_status'] == 'captured']
print("\nFiltered Data (Successful Payments):\n", df_filtered)

# Group by course_name
df_grouped = df.groupby('course_name')
print("\nGrouped Data:\n", df_grouped.size())

# Merging Course_name and Course_type dataframes
courses = df["course_name"].unique()

df2 = pd.DataFrame({
    "course_name": courses,
    "course_type": ["Technical"] * len(courses)
})
df_merged = pd.merge(df, df2, on="course_name")

print(df_merged.head(10))

# Pivot Table: Payment Status vs Course
df_pivot = pd.pivot_table(
    df,
    values='price',
    index='payment_status',
    columns='course_name',
    aggfunc='sum',
    fill_value=0
)

print("\nPivot Table (Payment Status vs Course):")
print(df_pivot)

# Exporting to CSV and Parquet
df.to_csv("Day2_Cleaned_Data.csv", index=False)
df.to_parquet("Day2_Cleaned_Data.parquet")


csv_size = os.path.getsize("Day2_Cleaned_Data.csv")
print("CSV size:", csv_size, "bytes")

parquet_size = os.path.getsize("Day2_Cleaned_Data.parquet")
print("Parquet size:", parquet_size, "bytes")