#1.DATA EXPLORATION 

import pandas as pd

import os


path = r"C:\SEMESTER\COMP DATAWAREHOUSE\Project"


filename = "Bicycle_Thefts.csv"


fullpath = os.path.join(path, filename)


data_maina = pd.read_csv(fullpath)


print("\n-----------Column names:")
print(data_maina.columns)


print("\n----------Shape:")
print(data_maina.shape)


print("\n--------------Summary statistics:")
print(data_maina.describe())


print("\n-----------------Data types:")
print(data_maina.dtypes)


print("\n------------------First five records:")
print(data_maina.head())


print("\n-----------DATASET:")
print(data_maina.info())


print("\n-------------MISSING VALUES:")
print(data_maina.isnull().sum())


print("\n----------- UNIQUE VALUES:")
print(data_maina.nunique())


print("\n-------- SAMPLE CATEGORICAL VALUES:")

categorical_columns = data_maina.select_dtypes(include=['object']).columns

for column in categorical_columns:
    print(f"\n{column}")
    print(data_maina[column].value_counts().head(10))
    
#B STATISTIC ASSESSMENTS

import numpy as np


print("\n--------Find Numerical columns:")
print(data_maina.select_dtypes(include=np.number).columns)  

print("\n-------------Calculate Mean:")
mean_value = data_maina.select_dtypes(include=np.number).mean()

print(mean_value)

print("\n---------------Calculate Median:")
median_values = data_maina.select_dtypes(include=np.number).median()

print(median_values)

print ("\n----------------Calculate Standard Deviation:")
std_values = data_maina.select_dtypes(include=np.number).std()

print(std_values)

print("\n----------------------Correlation analysis")
correlation =data_maina.select_dtypes(include=np.number).corr()

print(correlation)

print("\n-----------Minimum ")