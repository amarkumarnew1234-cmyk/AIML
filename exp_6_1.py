import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("Program Started")

# -------------------------------------------------
# Step 1: Load Dataset
# -------------------------------------------------
df = pd.read_csv(
    "employee1.csv",   # ✅ file name updated
    na_values=["", "NA", "NAN", "null"]
)

print("\nOriginal Dataset")
print(df)

# -------------------------------------------------
# Step 2: Remove Duplicate Rows
# -------------------------------------------------
df.drop_duplicates(inplace=True)

# -------------------------------------------------
# Step 3: Handle Missing Values
# -------------------------------------------------
num_cols = ["Age", "Experience", "Salary", "PerformanceScore",
            "ProjectsHandled", "WorkHoursPerWeek"]

for col in num_cols:
    if col in df.columns:   # ✅ safety check
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

if "Department" in df.columns:
    df["Department"] = df["Department"].fillna(df["Department"].mode()[0])

if "PromotionStatus" in df.columns:
    df["PromotionStatus"] = df["PromotionStatus"].fillna("No")

print("\nMissing values after handling")
print(df.isnull().sum())

# -------------------------------------------------
# Step 4: Logical Validation
# -------------------------------------------------
if "PerformanceScore" in df.columns:
    df["PerformanceScore"] = np.where(
        df["PerformanceScore"] > 100, 100, df["PerformanceScore"]
    )

if "WorkHoursPerWeek" in df.columns:
    df["WorkHoursPerWeek"] = np.where(
        df["WorkHoursPerWeek"] > 70, 70, df["WorkHoursPerWeek"]
    )

# -------------------------------------------------
# Step 5: Outlier Detection (IQR Method)
# -------------------------------------------------
print("\nHandling Outliers using IQR")

for col in num_cols:
    if col in df.columns:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])

# -------------------------------------------------
# Step 6: Feature Engineering
# -------------------------------------------------
if "Salary" in df.columns and "ProjectsHandled" in df.columns:
    df["SalaryPerProject"] = df["Salary"] / df["ProjectsHandled"]

if "Experience" in df.columns:
    df["ExperienceLevel"] = pd.cut(
        df["Experience"],
        bins=[0,2,5,10,40],
        labels=["Beginner","Intermediate","Senior","Expert"]
    )

# -------------------------------------------------
# Step 7: Encoding Categorical Variables
# -------------------------------------------------
df_encoded = pd.get_dummies(df, drop_first=True)

# -------------------------------------------------
# Step 8: Visualization
# -------------------------------------------------
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
sns.boxplot(data=df.select_dtypes(include="number"))
plt.title("After Outlier Handling")

plt.subplot(1,2,2)
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()

# -------------------------------------------------
# Step 9: Final Dataset
# -------------------------------------------------
print("\nFinal Cleaned Dataset")
print(df)

print("\nEncoded Dataset for ML")
print(df_encoded.head())

print("\nProgram Completed Successfully")
