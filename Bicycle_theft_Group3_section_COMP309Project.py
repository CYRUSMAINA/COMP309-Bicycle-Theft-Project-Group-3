#1.DATA EXPLORATION 

import pandas as pd

import os

# Folder containing the CSV file
path = r"C:\SEMESTER\COMP DATAWAREHOUSE\Project"

# CSV filename
filename = "Bicycle_Thefts.csv"

# Combine folder and filename
fullpath = os.path.join(path, filename)

# Load the dataset
data_maina = pd.read_csv(fullpath)

# a. Display column names
print("Column names:")
print(data_maina.columns)

# b. Display shape
print("\nShape:")
print(data_maina.shape)

# c. Display summary statistics
print("\nSummary statistics:")
print(data_maina.describe())

# d. Display data types
print("\nData types:")
print(data_maina.dtypes)

# e. Display the first five records
print("\nFirst five records:")
print(data_maina.head())

# f. Dataset information
print("\n========== DATASET INFO ==========")
print(data_maina.info())

# G. Missing values
print("\n========== MISSING VALUES ==========")
print(data_maina.isnull().sum())

# H. Number of unique values
print("\n========== UNIQUE VALUES ==========")
print(data_maina.nunique())

# I. Sample values for categorical columns
print("\n========== SAMPLE CATEGORICAL VALUES ==========")

categorical_columns = data_maina.select_dtypes(include=['object']).columns

for column in categorical_columns:
    print(f"\n{column}")
    print(data_maina[column].value_counts().head(10))