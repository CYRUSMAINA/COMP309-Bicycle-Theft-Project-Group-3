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


#----------------------------------------#
#          MODEL BUILDING                #
#----------------------------------------#



from sklearn.linear_model import LogisticRegression 

#a)create LR model
logistic_model = LogisticRegression(
    max_iter = 1000,
    random_state = 42
    )
#train model
logistic_model.fit(
    X_train_smote,
    Y_train_smote)

print("Model trained.")
print("Here is the Model:")
print(logistic_model)

print(" Training records:")
print(X_train_smote.shape[0])

print("Feature:")
print(X_train_smote.shape[1])

#b)LR predictions

#Use test data to make prediction

Y_pred_logistic =logistic_model.predict(X_test)

print("Logistic Regressions Predictions")

print("Predicted first 10 values:")
print(Y_pred_logistic[:10])

print("Actual values of 10:")
print(Y_test[:10])

#c)LR model Score

from sklearn.metrics import accuracy_score

logistic_accuracy = accuracy_score(
    Y_test,
    Y_pred_logistic
    )

print("Logistic Regression Accuracy:")

print("Accuracy:",logistic_accuracy)
print("Accuracy % ",logistic_accuracy*100)

#LogisticR CONFUSION MATRIX

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#create confusion matrics
con_logistic = confusion_matrix (
    Y_test,
    Y_pred_logistic
    )
print("Logistic Regression CONFUSION MATRIX")
print(con_logistic)

#Dispaly ConfusionM
 
disp = ConfusionMatrixDisplay(
    confusion_matrix=con_logistic,
    display_labels=label_encoder.classes_
    )
disp.plot()

plt.title("Logistic Regression Confusion Matrix")
plt.show()

print(label_encoder.classes_)

#LOGISTIC REGRESSION ROC CURVE

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

#  test labels into binary 
Y_test_binary = label_binarize(
    Y_test,
    classes=[0, 1, 2]
)


Y_probability = logistic_model.predict_proba(X_test)

# Calculate ROC values
fpr = {}
tpr = {}
roc_auc = {}

for i in range(3):

    fpr[i], tpr[i], _ = roc_curve(
        Y_test_binary[:, i],
        Y_probability[:, i]
    )

    roc_auc[i] = auc(
        fpr[i],
        tpr[i]
    )

#AUC results

print("LOGISTIC REGRESSION ROC AUC:-")


print("RECOVERED AUC:", roc_auc[0])
print("STOLEN AUC:", roc_auc[1])
print("UNKNOWN AUC:", roc_auc[2])


# Plot ROC curves
plt.figure(figsize=(8, 6))

plt.plot(
    fpr[0],
    tpr[0],
    label="RECOVERED (AUC = %.2f)" % roc_auc[0]
)

plt.plot(
    fpr[1],
    tpr[1],
    label="STOLEN (AUC = %.2f)" % roc_auc[1]
)

plt.plot(
    fpr[2],
    tpr[2],
    label="UNKNOWN (AUC = %.2f)" % roc_auc[2]
)

# Random guess
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Guess"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Logistic Regression ROC Curve")

plt.legend()

plt.grid()

plt.show()


#DECISION TREE


from sklearn.tree import DecisionTreeClassifier

# Create D T model
decision_tree_model = DecisionTreeClassifier(
    random_state=42
)

# Train the Decision Tree using SMOTE
decision_tree_model.fit(
    X_train_smote,
    Y_train_smote
)


print("DECISION TREE MODEL TRAINED")


print("Model:")
print(decision_tree_model)

print("\nTraining records:")
print(X_train_smote.shape[0])

print("\nNumber of features:")
print(X_train_smote.shape[1])


# Make predictions using the unseen test data
Y_pred_tree = decision_tree_model.predict(X_test)

print(" PREDICTIONS:")

print("Predicted first 10 values:")
print(Y_pred_tree[:10])

print("Actual first 10 values:")
print(Y_test[:10])


from sklearn.metrics import accuracy_score

tree_accuracy = accuracy_score(
    Y_test,
    Y_pred_tree
)

print("\nDECISION TREE ACCURACY")

print("Accuracy:", tree_accuracy)

print("Accuracy %:", tree_accuracy * 100)


#DECISION TREEE CONFUSION MATRIX

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

tree_cm = confusion_matrix(
    Y_test,
    Y_pred_tree
)

print("DECISION TREE CONFUSION MATRIX")
print(tree_cm)

disp_tree = ConfusionMatrixDisplay(
    confusion_matrix=tree_cm,
    display_labels=label_encoder.classes_
)

disp_tree.plot()

plt.title("Decision Tree Confusion Matrix")
plt.show()


# DECISION TREE ROC CURVE


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Convert test labels into binary format
Y_test_binary_tree = label_binarize(
    Y_test,
    classes=[0, 1, 2]
)

# Get probability predictions from Decision Tree
Y_probability_tree = decision_tree_model.predict_proba(X_test)

# Calculate ROC values
fpr_tree = {}
tpr_tree = {}
roc_auc_tree = {}

for i in range(3):

    fpr_tree[i], tpr_tree[i], _ = roc_curve(
        Y_test_binary_tree[:, i],
        Y_probability_tree[:, i]
    )

    roc_auc_tree[i] = auc(
        fpr_tree[i],
        tpr_tree[i]
    )

#AUC results
print("DECISION TREE ROC AUC")


print("RECOVERED AUC:", roc_auc_tree[0])
print("STOLEN AUC:", roc_auc_tree[1])
print("UNKNOWN AUC:", roc_auc_tree[2])


# Plot ROC curves
plt.figure(figsize=(8, 6))

plt.plot(
    fpr_tree[0],
    tpr_tree[0],
    label="RECOVERED (AUC = %.2f)" % roc_auc_tree[0]
)

plt.plot(
    fpr_tree[1],
    tpr_tree[1],
    label="STOLEN (AUC = %.2f)" % roc_auc_tree[1]
)

plt.plot(
    fpr_tree[2],
    tpr_tree[2],
    label="UNKNOWN (AUC = %.2f)" % roc_auc_tree[2]
)

# Random guessing line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Guess"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Decision Tree ROC Curve")

plt.legend()

plt.grid()

plt.show()

#MODEL SERIALIZATION


import pickle
import os

# Project folder
project_path = r"C:\SEMESTER\COMP DATAWAREHOUSE\Project"

# Location where the model will be saved
model_path = os.path.join(
    project_path,
    "logistic_model.pkl"
)


with open(model_path, "wb") as file:
    pickle.dump(logistic_model, file)

print("\nLogistic Regression model saved.")
print("Model saved at:")
print(model_path)



# TEST DESERIALIZATION


with open(model_path, "rb") as file:
    loaded_logistic_model = pickle.load(file)

print("\nLogistic Regression model loaded successfully.")

# Make predictions using the loaded model
loaded_predictions = loaded_logistic_model.predict(X_test)

print("\nFirst 10 predictions from loaded model:")
print(loaded_predictions[:10])

print("\nFirst 10 predictions from original model:")
print(Y_pred_logistic[:10])


# SAVE PREPROCESSING OBJECTS


# Save numerical imputer
with open(os.path.join(project_path, "num_imputer.pkl"), "wb") as file:
    pickle.dump(num_imputer, file)

# Save categorical imputer
with open(os.path.join(project_path, "cat_imputer.pkl"), "wb") as file:
    pickle.dump(cat_imputer, file)

# Save One-Hot Encoder
with open(os.path.join(project_path, "encoder.pkl"), "wb") as file:
    pickle.dump(encoder, file)

# Save StandardScaler
with open(os.path.join(project_path, "standard_scaler.pkl"), "wb") as file:
    pickle.dump(standard_scaler, file)

# Save SelectKBest feature selector
with open(os.path.join(project_path, "selector.pkl"), "wb") as file:
    pickle.dump(selector, file)

# Save LabelEncoder
with open(os.path.join(project_path, "label_encoder.pkl"), "wb") as file:
    pickle.dump(label_encoder, file)

print("\nAll preprocessing objects saved.")

with open(os.path.join(project_path, "num_imputer.pkl"), "rb") as file:
    loaded_num_imputer = pickle.load(file)

with open(os.path.join(project_path, "cat_imputer.pkl"), "rb") as file:
    loaded_cat_imputer = pickle.load(file)

with open(os.path.join(project_path, "encoder.pkl"), "rb") as file:
    loaded_encoder = pickle.load(file)

with open(os.path.join(project_path, "standard_scaler.pkl"), "rb") as file:
    loaded_standard_scaler = pickle.load(file)

with open(os.path.join(project_path, "selector.pkl"), "rb") as file:
    loaded_selector = pickle.load(file)

with open(os.path.join(project_path, "label_encoder.pkl"), "rb") as file:
    loaded_label_encoder = pickle.load(file)

print("\nAll preprocessing objects loaded.")


# TEST SAVED MODEL WITH SAVED PREPROCESSING


# Load saved StandardScaler
with open(os.path.join(project_path, "standard_scaler.pkl"), "rb") as file:
    loaded_standard_scaler = pickle.load(file)

# Load saved SelectKBest
with open(os.path.join(project_path, "selector.pkl"), "rb") as file:
    loaded_selector = pickle.load(file)

# Load saved Logistic Regression model
with open(os.path.join(project_path, "logistic_model.pkl"), "rb") as file:
    loaded_logistic_model = pickle.load(file)

# X_test is currently already standardized and feature-selected,
# so for this verification we use the saved model directly.

saved_model_predictions = loaded_logistic_model.predict(X_test)


print("SAVED MODEL TEST")


print("First 10 predictions using saved model:")
print(saved_model_predictions[:10])

print("\nFirst 10 predictions using original model:")
print(Y_pred_logistic[:10])

# Check whether predictions are identical
if (saved_model_predictions == Y_pred_logistic).all():
    print("\nSUCCESS: Saved model predictions match original model.")
else:
    print("\nWARNING: Predictions do not match.")