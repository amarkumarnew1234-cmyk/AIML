# Step 1: Import libraries
from sklearn.datasets import make_classification # used to create artificial datasets, or used to handle imbalance data
from sklearn.model_selection import train_test_split # used to split the train and test data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# classifictaion report will tell about the presicion, recall, F1 score, and confusion matrix will tell about the what are the right and wrong predictions we have performed.
from imblearn.under_sampling import RandomUnderSampler # RandomUnderSampler randomly removes samples from the majority class in order to balance the dataset.
from collections import Counter # it is used to count the frequency of elements in a dataset.”

# Step 2: Create a sample imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=4,
                           n_redundant=2, n_clusters_per_class=1,
                           weights=[0.9, 0.1], flip_y=0, random_state=42)
print("X shape:", X.shape)
print("y shape:", y.shape)

print("Before undersampling:", Counter(y)) # display how many samples are there before undersampling

# Step 3: Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # “Split the dataset (X, y) into training and testing sets,
#using 70% of the data for training and 30% for testing, while keeping the split consistent using random_state = 42.”

# Step 4: Apply Random Under-Sampling
rus = RandomUnderSampler(random_state=42) # Use the under-sampler to remove random samples from the majority class
#in the training data and create a new balanced dataset (X_resampled, y_resampled)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train) # Use the RandomUnderSampler (rus) to remove random samples from the majority class in
#the training data (X_train, y_train),and store the balanced dataset in X_resampled and y_resampled.”

print("After undersampling:", Counter(y_resampled)) # Stores the new balanced (resampled) training data after removing extra samples from the majority class

# Step 5: Train a classifier
model = RandomForestClassifier(random_state=42) # create a random forest model  This step creates a Random Forest classifier and
#trains it on the balanced dataset obtained after under-sampling, so the model learns patterns from both classes equally.
model.fit(X_resampled, y_resampled) #Trains the Random Forest model on the balanced data so it learns patterns for classification

# Step 6: Predict on test data
y_pred = model.predict(X_test)  # provide predictions or Use the trained Random Forest model to predict the output (y values) for the unseen test data

# Step 7: Evaluate performance
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred)) # used to give presiocion , Recall and F1 score
import numpy as np

# Total points in train set
print("X_train shape:", X_train.shape)

# Total points after under-sampling
print("X_resampled shape:", X_resampled.shape)

# Check if resampled data are subset of X_train
common = np.isin(X_train, X_resampled).all()
print("Same data points?", common)

