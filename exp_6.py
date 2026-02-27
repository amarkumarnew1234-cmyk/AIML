import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("Program Started")

# -------------------------------------------------
# Step 1: Load dataset
# -------------------------------------------------
df = pd.read_excel(
    "students_data1.xlsx",
    na_values=["", "NA", "NAN", ","]
)

print("\nOriginal Dataset")
print(df)

# -------------------------------------------------
# Step 2: Check missing values
# -------------------------------------------------
print("\nMissing values before handling")
print(df.isnull().sum())

# -------------------------------------------------
# Step 3: Handle missing values (Mean Imputation)
# -------------------------------------------------
for col in ["Age", "Marks", "Attendance"]:
    if df[col].isnull().sum() > 0:
        mean_value = df[col].mean()
        df[col] = df[col].fillna(mean_value)

print("\nMissing values after handling")
print(df.isnull().sum())

# -------------------------------------------------
# Step 4: Outlier Detection & Handling (IQR Method)
# -------------------------------------------------
print("\nHandling Outliers using IQR Method")

for col in ["Age", "Marks", "Attendance"]:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"\nColumn: {col}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Upper Bound: {upper_bound}")

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print("Outliers Detected:")
    print(outliers[[col]])

    # Capping outliers
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# -------------------------------------------------
# Step 5: Visualization (Before & After)
# -------------------------------------------------
df_original = pd.read_excel(
    "students_data1.xlsx",
    na_values=["", "NA", "NAN", ","]
)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
sns.boxplot(data=df_original.select_dtypes(include="number"))
plt.title("Before Outlier Handling")

plt.subplot(1,2,2)
sns.boxplot(data=df.select_dtypes(include="number"))
plt.title("After Outlier Handling")

plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 6: Final Dataset
# -------------------------------------------------
print("\nFinal Preprocessed Dataset")
print(df)

print("\nProgram Completed Successfully")
