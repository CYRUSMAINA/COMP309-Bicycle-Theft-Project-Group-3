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

plt.figure(figsize=(8,6))

plot_data = data_maina[["BIKE_COST", "BIKE_SPEED"]].dropna()


sns.regplot(
    data=data_maina,
    x="BIKE_COST",
    y="BIKE_SPEED",
    ci=None,
    scatter_kws={"alpha":0.4},
    line_kws={"color":"red"}
)

plt.title("Relationship Between Bicycle Cost and Bicycle Speed")

plt.xlabel("Bike Cost ($)")
plt.ylabel("Bike Speed (Number of Gears)")

plt.show()


#=================================================================
# b) DATA MODELING (PREDICT OUTCOME OF A BIKE THEFT USING INFOR) #
#=================================================================


print("b) DATA MODELING (PREDICT OUTCOME OF A BIKE THEFT USING INFOR)")


Y = data_maina["STATUS"]

features = [

"OCC_YEAR",
"OCC_MONTH",
"OCC_DOW",
"OCC_DAY",
"OCC_DOY",
"OCC_HOUR",

"REPORT_YEAR",
"REPORT_MONTH",
"REPORT_DOW",
"REPORT_DAY",
"REPORT_DOY",
"REPORT_HOUR",

"DIVISION",
"LOCATION_TYPE",
"PREMISES_TYPE",

"BIKE_MAKE",
"BIKE_TYPE",
"BIKE_SPEED",
"BIKE_COLOUR",
"BIKE_COST",

"PRIMARY_OFFENCE",


"NEIGHBOURHOOD_158",
"NEIGHBOURHOOD_140",

"LONG_WGS84",
"LAT_WGS84"

]

X=data_maina[features].copy()

print(X.head())

print("\n Shape:-")
print(X.shape)

print("\n Missing Values:-")
print(X.isnull().sum())

print("Numerical Features: ")
numerical_features = X.select_dtypes(
    include=["int64","float64"]
    ).columns
print(numerical_features)


print("\nCategorical Features: ")
categorical_features = X.select_dtypes(
    include=["object"]
    ).columns
print(categorical_features)


from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="median")
X[numerical_features] = num_imputer.fit_transform(X[numerical_features])

cat_imputer = SimpleImputer(strategy="most_frequent")
X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])

print(X.isnull().sum()[X.isnull().sum() > 0])


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
    )

encoded_data =encoder.fit_transform(X[categorical_features])

#convert encoded data to dataframe
import pandas as pd
 
encoded_df=pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(categorical_features),
    index=X.index
    )

X=X.drop(columns=categorical_features)

X=pd.concat(
    [X,encoded_df],
    axis=1 
    )
print("-----------------------------------------------------------")
print(X.dtypes)
print(X.select_dtypes(include=["object"]).columns)

#encode target (Y) variables
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
Y_encoded=label_encoder.fit_transform(Y)

print("Classes Encoded: ")
print(label_encoder.classes_)

print("10 Encoded Target Values: ")
print(Y_encoded[:10])

#standardization

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()

X_standardized = standard_scaler.fit_transform(X) 

X_standardized=pd.DataFrame(
    X_standardized,
    columns=X.columns,
    index= X.index)

print(X_standardized.head())
print("Mean of standardized features:")
print(X_standardized.mean().head())

print("\nStandard deviation of standardized features:")
print(X_standardized.std().head())

#reduce 14411 features to 100 features

from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(
    score_func=f_classif,
    k=100
)

X_selected = selector.fit_transform(
    X_standardized,
    Y_encoded
)


print("Before features:")
print(X_standardized.shape)

print("\nAfter features:")
print(X_selected.shape)


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X_selected,
    Y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=Y_encoded
)

print("Train data:")
print(X_train.shape)

print("Tested data:")
print(X_test.shape)

#handle imbalanced dataset
from imblearn.over_sampling import SMOTE

smote=SMOTE(
    random_state=42
    )
X_train_smote,Y_train_smote = smote.fit_resample(
    X_train,
    Y_train)


print("Before:")
print(pd.Series(Y_train).value_counts())

print("\nAfter:")
print(pd.Series(Y_train_smote).value_counts())

