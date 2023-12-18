import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from models import logistic_regression, naive_bayes, random_forest
from plots import correlation_plots

from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA


# Seed
RANDOM_STATE = 3

# Import dataset
data = pd.read_csv("breast-cancer.csv")

# Feature selection
data = data.drop(columns=['id']) # The id column is not useful information
data = data.replace({'B':0,'M':1}) # One-hot encode labels

y = data['diagnosis']
X = data.drop(columns=['diagnosis']) # Remove target label from features

# Split into training (80%), testing(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Print info about data
#print(X.describe())

correlation_plots(data)

print('Logistic Regression:' )
logistic_regression(X_train, y_train, X_test, y_test)

print('Naive Bayes: ')
naive_bayes(X_train, y_train, X_test, y_test)

print('Random Forest: ')
random_forest(X_train, y_train, X_test, y_test)


