import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Import dataset
data = pd.read_csv("breast-cancer.csv")
y = data['diagnosis'].replace({'B':0,'M':1}) # One-hot encode labels
X = data.drop(columns=['diagnosis']) # Remove target label from features

# Feature selection
X = X.drop(columns=['id']) # The id column is not useful information

# Split into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=3)

# Print info about data
print(X.describe())
