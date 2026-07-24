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
minimum=data_maina.select_dtypes(include=np.number).min()
print(minimum)

print("\n------------------maximum")
maximum=data_maina.select_dtypes(include=np.number).max()
print(maximum)


#C. MISSING VALUE

print("------------Missing Value Percentages")

missing_per = (data_maina.isnull().sum()/ len(data_maina))*100
print(missing_per.round(2))


print ("\n------------Columns with Missing Values ")

missing_v =data_maina.isnull().sum()
missing_v = missing_v[missing_v > 0]

print(missing_v)

print ("\n----------negatitive bike cost")
print((data_maina["BIKE_COST"] < 0).sum())

print("\n-----------latitude")
print((data_maina["LAT_WGS84"] == 0).sum())

print("\n-----------Longititude")
print((data_maina["LONG_WGS84"] == 0).sum())

#D GRAPH & VISUALIZATION

import matplotlib.pyplot as plt
import seaborn as sns

#Histogram of bicycle cost
plt.figure(figsize=(8,5))

plt.hist(data_maina["BIKE_COST"], bins=30)

plt.title("Distribution of Bicycle Costs")

plt.xlabel("Bike Cost ($)")

plt.ylabel("Number of Bicycles")

plt.show()


#bicycle type stolen the most(count plot)
plt.figure(figsize=(10,6))

sns.countplot(
    data=data_maina,
    x="BIKE_TYPE"
)

plt.title("Number of Bicycle stolen the most by Bike Type")

plt.xticks(rotation=45)

plt.show()

#Bicycle theft by year(count plot)

plt.figure(figsize=(8,5))

sns.countplot(
    data=data_maina,
    x="REPORT_YEAR"
)

plt.title("Bicycle Thefts by Report Year")

plt.show()


#Top 10 Neighbourhoods with Bicycle Thefts

plt.figure(figsize=(12,6))

top_neighbourhoods = data_maina["NEIGHBOURHOOD_158"].value_counts().head(10)

sns.barplot(
    x=top_neighbourhoods.values,
    y=top_neighbourhoods.index
)

plt.title("Top 10 Neighbourhoods with Bicycle Thefts")

plt.xlabel("Number of Bicycle Thefts")

plt.ylabel("Neighbourhood")

plt.show()

# Correlation Heatmap


# Create list of numerical columns to analyze
corr_columns = [
    "OCC_YEAR",
    "OCC_DAY",
    "OCC_DOY",
    "OCC_HOUR",
    "REPORT_YEAR",
    "REPORT_DAY",
    "REPORT_DOY",
    "REPORT_HOUR",
    "BIKE_SPEED",
    "BIKE_COST",
    "LAT_WGS84",
    "LONG_WGS84"
]


plt.figure(figsize=(12,8))

corr_matrix = data_maina[corr_columns].corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Bicycle Theft Data")

plt.show()

#Missing Values Bar Chart

missing_values = data_maina.isnull().sum()

missing_values = missing_values[missing_values > 0]


plt.figure(figsize=(10,6))


sns.barplot(
    x=missing_values.values,
    y=missing_values.index
)


plt.title("Missing Values by Column")

plt.xlabel("Number of Missing Values")

plt.ylabel("Column Name")


plt.show()




