import pandas as pd
import numpy as np

df = pd.read_csv("Day3_Data.csv")
print("Data Loaded Successfully")
print(df)

# Data inspection
print("Data Inspection")
print(df.info()) # display information about the DataFrame
print(df.describe()) # Display summary statistics
print(df.isnull().sum())  # Count of missing values in each column
print(df.shape) # display the shape of the DataFrame

# Using the dropna() method
df1=df.dropna()  # Show rows with missing values removed
print("Rows with missing values removed:")
print(df1)

df2=df.dropna(axis=1,how='any') #Show columns with missing values removed
print("Columns with missing values removed:")
print(df2)

# Using the fillna() method
df3=df.fillna(0) # Fill missing values with 0
print("Fill missing values with 0:")
print(df3)

df4=df.ffill() # Fill missing values with the previous value
print("Forward Fill:")
print(df4)

df5=df.bfill()  # Fill missing values with the next value
print("Backward Fill:")
print(df5)

# Using the drop_duplicates() method

df6=df.drop_duplicates() # Show rows with duplicate values removed
print("Rows with duplicate values removed:")
print(df6)

df7=df.drop_duplicates(subset=['Salary']) # Show rows with duplicate values removed based on a Salary column
print("Rows with duplicate values removed based on Salary:")
print(df7)

 # Using the replace() method
df8=df.replace(np.nan, 0) # Replace NaN values with 0
print("Replace NaN values with 0:")
print(df8)

df9=df.replace({"Salary":{-3000 : 3000}}) # Replace -3000 with 3000 in the Salary column
print("Replace -3000 with 3000 in the Salary column:")
print(df9)

df10 = df.copy()
df10["Experience"] = df["Experience"].replace("three", 3) # Replace 'three' with 3 in the Experience column
print("Replace 'three' with 3 in the Experience column:")
print(df10)

# Using the astype() method
df11=df.astype({"Experience":'int'}) # Change the data type of the Experience column to int
print("Change the data type of the Experience column to int:")
print(df11)

print(df.info()) 
