# Step 1: Import libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# Step 2: Create imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=4,
                           n_redundant=2, n_clusters_per_class=1,
                           weights=[0.9, 0.1], flip_y=0, random_state=42)

print("Before oversampling:", Counter(y))

# Step 3: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Apply Random Over-Sampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

print("After oversampling:", Counter(y_resampled))

# Step 5: Train a classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))