# import requried libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset
df = pd.read_csv("Titanic-Dataset.csv")
print(df)

# explore the data structure
print(df.head())
print(df.tail())

# shape
print(df.shape)

# column names
print(df.columns)

# data types
print(df.dtypes)

# dataset info
print(df.info())

# statistic summary
print(df.describe())

# check missing values
print(df.isnull().sum())

# visualize missing values
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values")
plt.show()

# check dublicated values
df = df.drop_duplicates()
print(df)

# explore numerical variables
print(df.describe())

plt.figure(figsize=(6, 4))
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Eplore categorical variable
sns.countplot(x="Sex", data=df)
plt.show()

# passenger class
sns.countplot(x="Pclass", data=df)
plt.show()

# indentify trends and patterns
# survival by gender
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()

# servival by passenger class
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.show()

# Age distribution by servival plot
sns.boxplot(x="Survived", y="Age", data=df)
plt.show()

# Detect outliers
sns.boxplot(y=df["Fare"])
plt.title("Fare Outliers")
plt.show()

# correlation analysis
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.show()

# Test Hypothesis 1
# females servived more than males
df.groupby("Sex")["Survived"].mean()

# Hypothesis 2
# first- class passenger servived more
df.groupby("Pclass")["Survived"].mean()

# Hypothesis 3
# high fear increases servival chance
sns.scatterplot(x="Fare", y="Survived", data=df)
plt.show()

# detect data issues
# missing values
df.isnull().sum()

# dublicated sum
df.duplicated().sum()

# outliers
sns.boxplot(df["Age"])
plt.show()

sns.boxplot(df["Fare"])
plt.show()

# data types

df.dtypes
