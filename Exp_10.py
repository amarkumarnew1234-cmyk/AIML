# Experiment 10: Data Visualization using Matplotlib and Seaborn 
# Fully corrected version for Python 3.14 
# Import required libraries 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
# Load the built-in Iris dataset from Seaborn 
iris = sns.load_dataset('iris') 
# Display first few records 
print("Sample Data:\n", iris.head()) 
# --------------------------- 
# 1. Line Plot using Matplotlib 
plt.figure(figsize=(6,4)) 
plt.plot(iris['sepal_length'], color='blue', label='Sepal Length') 
plt.title('Line Plot - Sepal Length') 
plt.xlabel('Index') 
plt.ylabel('Sepal Length (cm)') 
plt.legend() 
plt.show() 
# --------------------------- 
# 2. Bar Plot using Seaborn (corrected) 
plt.figure(figsize=(6,4)) 
sns.barplot(x='species', y='sepal_length', data=iris)  # removed palette to avoid warning 
plt.title('Bar Plot - Average Sepal Length by Species') 
plt.show() 
# --------------------------- 
# 3. Histogram using Matplotlib 
plt.figure(figsize=(6,4)) 
plt.hist(iris['petal_length'], bins=15, color='orange', edgecolor='black') 
plt.title('Histogram - Petal Length Distribution') 
plt.xlabel('Petal Length (cm)') 
plt.ylabel('Frequency') 
plt.show() 
# --------------------------- 
# 4. Scatter Plot using Seaborn 
plt.figure(figsize=(6,4)) 
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=iris) 
plt.title('Scatter Plot - Sepal vs Petal Length') 
plt.show() 
# --------------------------- 
# 5. Box Plot using Seaborn (corrected) 
plt.figure(figsize=(6,4)) 
sns.boxplot(x='species', y='petal_width', data=iris)  # removed palette to avoid warning 
plt.title('Box Plot - Petal Width by Species') 
plt.show() 
# --------------------------- 
# 6. Heatmap using Seaborn (corrected) 
plt.figure(figsize=(6,4)) 
sns.heatmap(iris.select_dtypes(include='number').corr(),  # numeric columns only 
annot=True, cmap='coolwarm', linewidths=0.5) 
plt.title('Heatmap - Feature Correlation') 
plt.show() 
# --------------------------- 
# 7. Pair Plot using Seaborn 
sns.pairplot(iris, hue='species') 
plt.suptitle('Pair Plot - All Features', y=1.02) 
plt.show()