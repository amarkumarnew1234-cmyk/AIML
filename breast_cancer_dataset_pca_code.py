import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------
# Step 1: Dataset Load karna
# ---------------------------------------

# sklearn se dataset load kar rahe hain
data = load_breast_cancer()

# dataframe bana rahe hain
df = pd.DataFrame(data.data, columns=data.feature_names)

# target column add kar rahe hain
df["Target"] = data.target

# data ka first look
print("Original Dataset:")
print(df.head())

# ---------------------------------------
# Step 2: Input aur Output alag karna
# ---------------------------------------

# Target (label) alag kar rahe hain
y = df["Target"]

# features (input) le rahe hain
X = df.drop("Target", axis=1)

# ---------------------------------------
# Step 3: Data Scaling (Standardization)
# ---------------------------------------

# scaler object bana rahe hain
scaler = StandardScaler()

# data ko scale kar rahe hain
X_scaled = scaler.fit_transform(X)

# scaled data ka shape check
print("Scaled Data Shape:", X_scaled.shape)

# ---------------------------------------
# Step 4: PCA Apply karna (All Components)
# ---------------------------------------

# PCA object
pca = PCA()

# PCA apply kar rahe hain
X_pca = pca.fit_transform(X_scaled)

# explained variance ratio
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# cumulative variance
print("Cumulative Variance:")
print(np.cumsum(pca.explained_variance_ratio_))

# ---------------------------------------
# Step 5: 2 Principal Components lena
# ---------------------------------------

pca2 = PCA(n_components=2)

# reduce kar rahe hain data ko
X_pca2 = pca2.fit_transform(X_scaled)

# ---------------------------------------
# Step 6: PCA DataFrame banana
# ---------------------------------------

df_pca = pd.DataFrame(X_pca2, columns=["PC1","PC2"])

# target wapas add kar rahe hain
df_pca["Target"] = y

# readable labels bana rahe hain
df_pca["Target"] = df_pca["Target"].map({0:"Malignant", 1:"Benign"})

print("PCA Data:")
print(df_pca.head())

# ---------------------------------------
# Step 7: Visualization
# ---------------------------------------

plt.figure(figsize=(8,6))

sns.scatterplot(x="PC1", y="PC2", hue="Target", data=df_pca)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Breast Cancer Dataset")

plt.show()

# ---------------------------------------
# Step 8: Final Shapes check
# ---------------------------------------

print("Original Shape:", X.shape)
print("After PCA Shape:", X_pca2.shape)