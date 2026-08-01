#=========================================================================
# BICYCLE THEFT PREDICTION - COMP309 GROUP PROJECT #2
#=========================================================================

# ---------------------- ALL IMPORTS ----------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
# -------------------------------------------------------------------------


#1.DATA EXPLORATION

#path = r"C:\SEMESTER\COMP DATAWAREHOUSE\Project"

path = r"C:\comp309\COMP309-Bicycle-Theft-Project-Group-3" #mohmedjuber

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


#=================================================================
# b) DATA MODELING (PREDICT OUTCOME OF A BIKE THEFT USING INFOR) #
#=================================================================


print("b) DATA MODELING (PREDICT OUTCOME OF A BIKE THEFT USING INFOR)")

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
"BIKE_MODEL",
"BIKE_TYPE",
"BIKE_SPEED",
"BIKE_COLOUR",
"BIKE_COST",

"PRIMARY_OFFENCE",

"HOOD_158",
"NEIGHBOURHOOD_158",
"HOOD_140",
"NEIGHBOURHOOD_140",

"LONG_WGS84",
"LAT_WGS84"

]

# .copy() ensures x is independent, so cleaning/encoding
# steps below don't risk altering the original dataframe
x = data_maina[features].copy()

print(x.head())

print("\n Shape:-")
print(x.shape)

print("\n Missing Values:-")
print(x.isnull().sum())


# --------------------------------------------------------------
# TARGET VARIABLE: turn STATUS into binary (returned vs not)
# The project asks us to predict whether a bike is "likely to be
# returned or not". RECOVERED = returned = 1, STOLEN/UNKNOWN = 0.
# --------------------------------------------------------------
Y = data_maina["STATUS"].apply(lambda s: 1 if s == "RECOVERED" else 0)

print("\nClass balance BEFORE any balancing:")
print(Y.value_counts())
print(Y.value_counts(normalize=True).round(4) * 100, "%")


# --------------------------------------------------------------
# SEPARATE NUMERIC vs CATEGORICAL COLUMNS
# --------------------------------------------------------------
numeric_cols = x.select_dtypes(include=np.number).columns.tolist()
categorical_cols = x.select_dtypes(include="object").columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)


# --------------------------------------------------------------
# HANDLE MISSING DATA
# --------------------------------------------------------------
# Numeric: fill with median (robust to outliers like BIKE_COST)
for col in numeric_cols:
    if x[col].isnull().sum() > 0:
        median_val = x[col].median()
        x[col] = x[col].fillna(median_val)
        print(f"Filled {col} missing values with median = {median_val}")

# Categorical: fill with the literal string "UNKNOWN" so it's its own category
for col in categorical_cols:
    if x[col].isnull().sum() > 0:
        x[col] = x[col].fillna("UNKNOWN")
        print(f"Filled {col} missing values with 'UNKNOWN'")

print("\nRemaining missing values:\n", x.isnull().sum())


# --------------------------------------------------------------
# ENCODE CATEGORICAL COLUMNS TO NUMBERS
# --------------------------------------------------------------
# Models can't use text directly. We use LabelEncoder because several
# of these columns (BIKE_MODEL, PRIMARY_OFFENCE, NEIGHBOURHOOD_158, etc.)
# have very high cardinality - one-hot encoding would blow up the
# number of columns.
label_encoders = {}  # keep these! needed later to encode new incoming
                      # API requests the same way.

for col in categorical_cols:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col].astype(str))
    label_encoders[col] = le

print(x.head())
print("\nShape after encoding:", x.shape)


# --------------------------------------------------------------
# TRAIN / TEST SPLIT -- before scaling and before balancing
# --------------------------------------------------------------
# Splitting first prevents information from the test set leaking into
# training via the scaler or the balancing step, which would make
# evaluation results artificially optimistic / invalid.
X_train, X_test, y_train, y_test = train_test_split(
    x, Y,
    test_size=0.30,
    random_state=42,
    stratify=Y   # keeps the same class ratio in train and test
)

print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)
print("\nTrain class balance:\n", y_train.value_counts())
print("\nTest class balance:\n", y_test.value_counts())


# --------------------------------------------------------------
# NORMALIZE / STANDARDIZE NUMERIC FEATURES
# --------------------------------------------------------------
# StandardScaler transforms each numeric column to mean=0, std=1, so
# large-range columns (e.g. BIKE_COST) don't dominate over small-range
# ones (e.g. OCC_HOUR). Fit on X_train ONLY, then transform X_test with
# that same fitted scaler - never fit on test data.
scaler = StandardScaler()

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

print("\nScaled numeric training columns summary:")
print(X_train[numeric_cols].describe())


# --------------------------------------------------------------
# BALANCE THE TRAINING DATA (SMOTE)
# --------------------------------------------------------------
# The training set is extremely imbalanced (~1% RECOVERED). A model
# trained on this as-is will just predict "not returned" every time and
# still score high accuracy while being useless. We oversample the
# minority class in the TRAINING set only - the test set is left with
# its real-world class proportions so evaluation stays honest.
print("\nBefore SMOTE:", y_train.value_counts().to_dict())

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("After SMOTE:", y_train_balanced.value_counts().to_dict())


# --------------------------------------------------------------
# SANITY CHECK - what to carry forward into model training
# --------------------------------------------------------------
print("\nUse these going forward for training your classifiers:")
print("X_train_balanced shape:", X_train_balanced.shape)
print("y_train_balanced distribution:\n", y_train_balanced.value_counts())
print("X_test shape (untouched, original distribution):", X_test.shape)
print("y_test distribution:\n", y_test.value_counts())

# Keep label_encoders and scaler - pickle these alongside your trained
# model in the deployment step, so the Flask API can transform new
# incoming feature values the exact same way before predicting.