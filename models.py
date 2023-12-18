from plots import ROC_curve, confusion_mat

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def random_forest(X_train, y_train, X_test, y_test):
    #Perform gridsearch
    model = GridSearchCV(RandomForestClassifier(), 
        param_grid={
            'criterion': ['gini', 'entropy'], 
            'max_depth': [None] 
            }, 
    cv=5, scoring="accuracy")

    #Fit the models
    model.fit(X_train, y_train)
    best_model = model.best_estimator_

    #Print best model
    print(best_model.get_params())

    #Print best score
    print('Best score: ', model.best_score_)

    # Predict the target values
    y_hat = best_model.predict(X_test)

    # tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
    print(pd.crosstab(pd.Series(y_test.to_numpy(), name='Actual'), pd.Series(y_hat, name='Predicted')), "\n")

    ROC_curve(model, X_test, y_test, "Random Forest")
    confusion_mat(model, y_hat, y_test, "Random Forest")


def naive_bayes(X_train, y_train, X_test, y_test):
    
    model = GridSearchCV(GaussianNB(), param_grid = {}, cv=5, scoring="accuracy")

    #Fit the models
    model.fit(X_train, y_train)
    best_model = model.best_estimator_

    #Print best model
    print(best_model.get_params())

    #Print best score
    print('Best score: ', model.best_score_)

    # Predict the target values
    y_hat = best_model.predict(X_test)

    # tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
    print(pd.crosstab(pd.Series(y_test.to_numpy(), name='Actual'), pd.Series(y_hat, name='Predicted')), "\n")

    ROC_curve(model, X_test, y_test, "Naive Bayes")
    confusion_mat(model, y_hat, y_test, "Naive Bayes")

def logistic_regression(X_train, y_train, X_test, y_test):

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

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
    y_hat = best_model.predict(X_test)

    # tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
    print(pd.crosstab(pd.Series(y_test.to_numpy(), name='Actual'), pd.Series(y_hat, name='Predicted')), "\n")

    ROC_curve(model, X_test, y_test, "Logistic Regression")
    confusion_mat(model, y_hat, y_test, "Logistic Regression")

