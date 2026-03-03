import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("students_data.csv")

print("Dataset Info")
print(df.info())

print("First Five Rows")
print(df.head())

print("Missing Values")
print(df.isnull().sum())

print("Summary Statistics")
print(df.describe())

print("Correlation Matrix")
print(df.corr(numeric_only=True))

df.hist(figsize=(10,6))
plt.suptitle("Histogram of Numerical Columns")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=df.iloc[:,0], y=df.iloc[:,1])
plt.title("Scatter Plot")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()