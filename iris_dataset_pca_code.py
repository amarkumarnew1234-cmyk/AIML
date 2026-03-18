import pandas as pd          # data read aur handle karne ke liye
import numpy as np           # numerical calculations ke liye
from sklearn.preprocessing import StandardScaler   # data scale karne ke liye
from sklearn.decomposition import PCA              # PCA apply karne ke liye
import matplotlib.pyplot as plt    # graph plot karne ke liye
import seaborn as sns              # better visualization

# dataset load kar rahe hain (file same folder me honi chahiye)
df = pd.read_csv("Iris.csv")

# Id column hata rahe hain kyunki wo useful nahi hai
df = df.drop("Id", axis=1)

# Species column alag store kar rahe hain (ye label hai)
y = df["Species"]

# baaki sirf features le rahe hain
X = df.drop("Species", axis=1)

# Standardization: sab values ko same scale pe la rahe hain
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA apply kar rahe hain (pehle saare components dekhte hain)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# har component kitna important hai wo print kar rahe hain
print("Explained variance ratio:", pca.explained_variance_ratio_)

# cumulative variance (total kitna data cover ho gaya)
print("Cumulative variance:", np.cumsum(pca.explained_variance_ratio_))

# ab sirf 2 components le rahe hain (graph ke liye)
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)

# PCA ke baad data ko dataframe bana rahe hain
pca_df = pd.DataFrame(data=X_pca2, columns=["PC1", "PC2"])

# Species column wapas add kar rahe hain
pca_df["Species"] = y

# scatter plot bana rahe hain (different species alag color me dikhenge)
sns.scatterplot(x="PC1", y="PC2", hue="Species", data=pca_df)

# axis labels
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# title
plt.title("PCA on Iris Dataset")

# graph show
plt.show()

# shapes check kar rahe hain
print("Original shape:", X.shape)
print("Scaled shape:", X_scaled.shape)
print("After PCA shape:", X_pca2.shape)