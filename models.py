import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix


def random_forest(X_train, y_train, X_val, y_val):
    #Perform gridsearch
    model = GridSearchCV(RandomForestClassifier(), param_grid={'max_depth': [None, 10, 5]}, 
    cv=5, scoring="accuracy")

    #Fit the models
    model.fit(X_train, y_train)
    best_model = model.best_estimator_

    #Print best model
    print(best_model.get_params())

    #Print best score
    print('Best score: ', model.best_score_)

    # Predict the target values
    y_hat = best_model.predict(X_val)

    # tn, fp, fn, tp = confusion_matrix(y_val, y_hat).ravel()
    print(pd.crosstab(pd.Series(y_val.to_numpy(), name='Actual'), pd.Series(y_hat, name='Predicted')), "\n")

    #print(y_val.to_numpy())
    #print(y_hat)


def naive_bayes(X_train, y_train, X_val, y_val):
    
    model = GridSearchCV(GaussianNB(), param_grid = {}, cv=5, scoring="accuracy")

    #Fit the models
    model.fit(X_train, y_train)
    best_model = model.best_estimator_

    #Print best model
    print(best_model.get_params())

    #Print best score
    print('Best score: ', model.best_score_)

    # Predict the target values
    y_hat = best_model.predict(X_val)

    # tn, fp, fn, tp = confusion_matrix(y_val, y_hat).ravel()
    print(pd.crosstab(pd.Series(y_val.to_numpy(), name='Actual'), pd.Series(y_hat, name='Predicted')), "\n")

    #print(y_val.to_numpy())
    #print(y_hat)


def logistic_regression(X_train, y_train, X_val, y_val):

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)

    # Find best hyperparameters
    model = GridSearchCV(LogisticRegression(solver="liblinear"), 
        param_grid={'penalty': ['l1', 'l2'], 'C': np.logspace(-2,2,5)}, 
        cv=5, 
        scoring="accuracy")

    # Fit the model to the training data
    model.fit(X_train, y_train)
    best_model = model.best_estimator_

    print(best_model.get_params())
    
    #Print best score
    print('Best score: ', model.best_score_)

    # Predict the target values
    y_hat = best_model.predict(X_val)

    # tn, fp, fn, tp = confusion_matrix(y_val, y_hat).ravel()
    print(pd.crosstab(pd.Series(y_val.to_numpy(), name='Actual'), pd.Series(y_hat, name='Predicted')), "\n")

    #print(y_val.to_numpy())
    #print(y_hat)