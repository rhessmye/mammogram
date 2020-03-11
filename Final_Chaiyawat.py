# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:37:56 2020

@author: User
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

### DATA PREPARATION ###
## Import Data 
df_mam = pd.read_csv(r'D:\_RADS602_Data_mining_Machine_Learning\Assignment_09Mar_Presentation\mammographic_masses_with_head.data')


## Delete record with missing value
df_mam = df_mam.apply(pd.to_numeric, errors='coerce')
df_mam.dtypes
df_mam=df_mam.dropna()

## Data checking
df_mam.describe()

pd.crosstab(index=df_mam['malig'], columns=df_mam['birads'])
pd.crosstab(index=df_mam['malig'], columns=df_mam['shape'])
pd.crosstab(index=df_mam['malig'], columns=df_mam['margin'])
pd.crosstab(index=df_mam['malig'], columns=df_mam['density'])

## There is (are) records with birads =55
## Remove abnormal value
df_mam = df_mam.drop(df_mam[df_mam.birads > 6].index)
## recheck the data again, and they are now acceptable
## thus now, there are 829 observations from originally 961 observations

## create dummies variable for using in Logistric Regrssion model
shape = pd.get_dummies(df_mam["shape"])
margin = pd.get_dummies(df_mam["margin"])
density = pd.get_dummies(df_mam["density"])

shape.columns =["s_round", "s_oval", "s_lobular", "s_irregular"]
margin.columns = ["m_circumscribed", "m_microlobulated", "m_obscured", "m_ill-defined", "m_spiculated"]
density.columns = ["d_high", "d_iso","d_low","d_fat" ]

## save the df_mam to df_mam_full for other use
df_mam_full = df_mam

## final dataframe with dummy variables
df_mam = pd.concat([df_mam,shape,margin,density], axis=1)
df_mam = df_mam.drop('shape', axis=1)
df_mam = df_mam.drop('margin', axis=1)
df_mam = df_mam.drop('density', axis=1)

## Create X, y as predictors and Dichotomous outcome variable, respectively
X=df_mam.drop('malig', axis=1)
y=df_mam['malig']

## Create training and testing dataset with ratio of 80:20
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.20)

## Export to cvs files
X_train.to_csv(r'D:/_RADS602_Data_mining_Machine_Learning/Assignment_09Mar_Presentation/X_train.csv')
X_test.to_csv(r'D:/_RADS602_Data_mining_Machine_Learning/Assignment_09Mar_Presentation/X_test.csv')
y_train.to_csv(r'D:/_RADS602_Data_mining_Machine_Learning/Assignment_09Mar_Presentation/y_train.csv')
y_test.to_csv(r'D:/_RADS602_Data_mining_Machine_Learning/Assignment_09Mar_Presentation/y_test.csv')

### END OF DATA PREPARATION ###


### LOGISTIC REGRESSION MODEL ###
## Model without tuning
clf = LogisticRegression()
clf.fit(X_train, y_train)

px = clf.predict_proba(X_test)
yhat = clf.predict(X_test)

confmat_transpose = confusion_matrix(y_test,yhat).transpose()
confmat = confusion_matrix(y_test,yhat)
# True Positives
TP = confmat[1, 1]
# True Negatives
TN = confmat[0, 0]
# False Positives
FP = confmat[0, 1]
# False Negatives
FN = confmat[1, 0]
print('accuracy:', (TP+TN)/float(TP+TN+FP+FN))
print('sensitivity:', TP/float(TP+FN))
print('specificity:', TN/float(TN+FP))


print('Results on the test set:')
print(classification_report(y_test, yhat))


### hyperparameter tuning ###
lr={}

#randomized search
logreg_hyparams={"C": [0.0001, 0.001, 0.01, 0.1, 1],
                 "solver": ["lbfgs", "liblinear","sag","saga"]}
lr["randomcv"]= RandomizedSearchCV(estimator=LogisticRegression(max_iter=10000, penalty="l2"),
                                      param_distributions = logreg_hyparams,
                                      cv=10,
                                      verbose=5)
lr["learn"]=lr["randomcv"].fit(X_train, y_train)
lr_fine_tuned ={"C": lr["learn"].best_estimator_.get_params()["C"],
                "solver": lr["learn"].best_estimator_.get_params()["solver"]}
print(lr_fine_tuned)
clf = LogisticRegression(penalty='l2',
                         C = 1, 
                         solver= 'liblinear',
                         random_state = 0)

clf.fit(X_train, y_train)

px = clf.predict_proba(X_test)
yhat2 = clf.predict(X_test)

confmat_transpose = confusion_matrix(y_test,yhat2).transpose()
confmat = confusion_matrix(y_test,yhat2)

TP = confmat[1, 1]
TN = confmat[0, 0]
FP = confmat[0, 1]
FN = confmat[1, 0]
print('accuracy:', (TP+TN)/float(TP+TN+FP+FN))
print('sensitivity:', TP/float(TP+FN))
print('specificity:', TN/float(TN+FP))

print('Results on the test set:')
print(classification_report(y_test, yhat2))


# GridSearch
logistic = linear_model.LogisticRegression()
# Create regularization penalty space
penalty = ['l1', 'l2']
# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)
# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

clf2 = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
best_model = clf2.fit(X_train, y_train)
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

yhat3 = best_model.predict(X_test)

confmat_transpose = confusion_matrix(y_test,yhat3).transpose()
confmat = confusion_matrix(y_test,yhat3)

TP = confmat[1, 1]
TN = confmat[0, 0]
FP = confmat[0, 1]
FN = confmat[1, 0]
print('accuracy:', (TP+TN)/float(TP+TN+FP+FN))
print('sensitivity:', TP/float(TP+FN))
print('specificity:', TN/float(TN+FP))

print('Results on the test set:')
print(classification_report(y_test, yhat3))

### END OF LOGISTIC REGRESSION MODEL ###

### RANDOM FOREST ###
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred_rd=clf.predict(X_test)

confmat_transpose = confusion_matrix(y_test,y_pred_rd).transpose()
confmat = confusion_matrix(y_test,y_pred_rd)

TP = confmat[1, 1]
TN = confmat[0, 0]
FP = confmat[0, 1]
FN = confmat[1, 0]
print('accuracy:', (TP+TN)/float(TP+TN+FP+FN))
print('sensitivity:', TP/float(TP+FN))
print('specificity:', TN/float(TN+FP))

print('Results on the test set:')
print(classification_report(y_test, y_pred_rd))

### END OF RANDOM FOREST ###

### MLP CLASSIFIER ###

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(max_iter=100)

parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train, y_train)

# Best paramete set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_true, y_pred_mlp = y_test , clf.predict(X_test)

confmat_transpose = confusion_matrix(y_test,y_pred_mlp).transpose()
confmat = confusion_matrix(y_test,y_pred_mlp)

TP = confmat[1, 1]
TN = confmat[0, 0]
FP = confmat[0, 1]
FN = confmat[1, 0]
print('accuracy:', (TP+TN)/float(TP+TN+FP+FN))
print('sensitivity:', TP/float(TP+FN))
print('specificity:', TN/float(TN+FP))

from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred_mlp))

### END OF MLP CLASSIFIER ###

















