#Arjun Verma
#July 2023

##Code to develop XGBoost model to predict clinically relevant postoperative pancreatic fistula following pancreaticoduodenectomy

###import necessary packages
import numpy as np
import pandas as import pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV, RFE
from xgboost import XGBClassifier
import pickle

###obtain training data with all candidate covariates from the ACS NSQIP
X_train (8597, 58)
y_train (8597, 1)

###perform hyperparameter tuning
params = {'max_depth': [1, 2, 3, 5, 7],'learning_rate': [0.1, 0.01, 0.001],'n_estimators': [20, 30, 40, 50, 75, 100, 200],'min_child_weight': [1, 3, 5],'gamma': [0, 0.1, 0.5],'subsample': [0.5, 0.8, 1],'colsample_bytree': [0.5, 0.8, 1]}
xgb_clf = XGBClassifier()
grid_search = GridSearchCV(xgb_clf, params, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

###retrieve best hyperparameters and create instance of model
grid_search.best_params_
xgb_tuned = XGBClassifier(colsample_bytree=1, gamma=0, learning_rate=0.1, max_depth=1, min_child_weight=1, n_estimators=200, subsample=1)

###perform recursive feature elimination
rfecv = RFECV(estimator=xgb_tuned, step=1, cv=5, scoring="roc_auc")
rfecv.fit(X_train, y_train)

###retrieve optimal number of features and identify best features
rfecv.n_features_
selector = RFE(xgb_tuned, n_features_to_select=rfecv.n_features_, step=1)
selector = selector.fit(X_train, y_train)

###create instance of final model and train using selected features
xgb_final = XGBClassifier(colsample_bytree=1, gamma=0, learning_rate=0.1, max_depth=1, min_child_weight=1, n_estimators=200, subsample=1)
X_train (8597, 29)
y_train (8597, 1)
xgb_final.fit(X_train, y_train)

###save model
filename = "xgb_export.pkl"
pickle.dump(xgb_final, open(filename, "wb"))
