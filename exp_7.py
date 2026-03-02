# ---------------------------------------
# Experiment 7: Feature Scaling & One Hot Encoding
# ---------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------
# Step 1: Read Excel Dataset
# ---------------------------------------
df = pd.read_excel("Student_data_1.xlsx")

df.columns = df.columns.str.strip()

print("Original Dataset:")
print(df)

# ---------------------------------------
# Step 2: Handle Missing Values
# ---------------------------------------
df["Age"] = df["Age"].fillna(df["Age"].mean())

# ---------------------------------------
# Step 3: Feature Scaling
# ---------------------------------------
scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler()

num_cols = ["Age", "Marks", "Attendance"]

df_minmax = pd.DataFrame(
    scaler_minmax.fit_transform(df[num_cols]),
    columns=[col + "_Norm" for col in num_cols]
)

df_standard = pd.DataFrame(
    scaler_standard.fit_transform(df[num_cols]),
    columns=[col + "_Std" for col in num_cols]
)

# ---------------------------------------
# Step 4: One Hot Encoding
# ---------------------------------------
encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(df[["Student"]])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(["Student"])
)

# ---------------------------------------
# Step 5: Combine All Data
# ---------------------------------------
df_final = pd.concat([df, df_minmax, df_standard, encoded_df], axis=1)

print("\nFinal Preprocessed Dataset:")
print(df_final)

# ---------------------------------------
# Step 6: Save Processed File
# ---------------------------------------
df_final.to_csv("Student_data_1_processed.csv", index=False)

print("\nProcessed file saved as 'Student_data_1_processed.csv'")