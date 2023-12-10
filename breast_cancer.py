import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from models import logistic_regression

from sklearn.model_selection import train_test_split


# Seed
RANDOM_STATE = 3

# Import dataset
data = pd.read_csv("breast-cancer.csv")
y = data['diagnosis'].replace({'B':0,'M':1}) # One-hot encode labels
X = data.drop(columns=['diagnosis']) # Remove target label from features

# Feature selection
X = X.drop(columns=['id']) # The id column is not useful information

# Split into training (70%), validation (20%), and testing (10%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.33, random_state=RANDOM_STATE)

# Print info about data
print(X.describe())


logistic_regression(X_train, y_train, X_val, y_val)