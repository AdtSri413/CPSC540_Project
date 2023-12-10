import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Seed
RANDOM_STATE = 3

# Import dataset
data = pd.read_csv("lung-cancer.csv")
y = data['LUNG_CANCER'].replace({'YES':0, 'NO':1}) # One-hot encode labels
X = data.drop(columns=['LUNG_CANCER']) # Remove target label from features

# Feature selection/transformation
X = X.replace({2:0, 'F':0, 'M':1}) # One-hot encode features. Let yes=0 and no=1

# Split into training (70%), validation (20%), and testing (10%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.33, random_state=RANDOM_STATE)

# Print info about data
print(X.describe())