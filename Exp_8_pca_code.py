import pandas as pd          # data handle karne ke liye
import numpy as np           # numerical operations ke liye
from sklearn.preprocessing import StandardScaler   # data ko scale karne ke liye
from sklearn.decomposition import PCA              # PCA apply karne ke liye
import matplotlib.pyplot as plt    # graph plot karne ke liye
import seaborn as sns              # better visualization ke liye

# dataset load kar rahe hain (file same folder me honi chahiye)
df = pd.read_csv("pca_ecommerce_customers.csv")

# CustomerID column ko hata diya kyunki wo useful nahi hai PCA ke liye
X = df.drop('CustomerID', axis=1)

# check kar rahe hain kaunse columns non-numeric hain (string type wale)
non_numeric_cols = X.select_dtypes(include=['object']).columns

# agar koi non-numeric column mila to usko drop kar denge
if len(non_numeric_cols) > 0:
    print("Dropping non-numeric columns:", list(non_numeric_cols))
    X = X.drop(columns=non_numeric_cols)

# Standardization: data ka mean 0 aur std deviation 1 kar dete hain
# PCA ke liye ye step bahut important hota hai
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA apply kar rahe hain (saare components ke saath)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# har component kitna variance explain kar raha hai wo print kar rahe hain
print("Explained variance ratio:", pca.explained_variance_ratio_)

# cumulative variance (total kitna data cover ho gaya) dekh rahe hain
print("Cumulative variance:", np.cumsum(pca.explained_variance_ratio_))

# ab sirf 2 principal components le rahe hain (visualization ke liye)
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)

# scatter plot bana rahe hain (2D me data dekhne ke liye)
sns.scatterplot(x=X_pca2[:,0], y=X_pca2[:,1])

# x-axis label
plt.xlabel("PC1")

# y-axis label
plt.ylabel("PC2")

# graph ka title
plt.title("Customer clusters after PCA")

# graph show karna
plt.show()

# original data ka shape (rows, columns)
print("Original shape:", X.shape)

# scaled data ka shape (same hi rahega)
print("Scaled shape:", X_scaled.shape)

# PCA lagane ke baad (all components)
print("Before PCA shape:", X_pca.shape)

# PCA ke baad sirf 2 components bache
print("After PCA shape:", X_pca2.shape)