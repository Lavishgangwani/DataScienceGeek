#logistic regression.py

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#creating a dataset
X,y = make_classification(
                n_samples=5000,
                n_classes=2,
                n_informative=2,
                n_features=10,
                random_state=23 
            )



#splitting the dataset
X_train , X_test ,y_train , y_test = train_test_split(X,
                                                      y,
                                                      test_size=0.3,
                                                      random_state=32)


#building a LogisticRegression model
model = LogisticRegression().fit(X_train,y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

#evaluating scores
accuracy = ("Accuracy Score is : ",accuracy_score(y_test,y_pred))
cm = ("Confusion Matrics : \n",confusion_matrix(y_test,y_pred))
cr = ("Classification Report :  \n",classification_report(y_test,y_pred))

## Hyperparameter Tuning and Cross Validation

#Using GridSearchCV  
model = LogisticRegression()
penalty = ['l1','l2','elasticnet']
c_values = [100,10,1.0,0.1,0.01,0.001,0.0001]
solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']

#Converting all the parameters into dict
params = dict(penalty=penalty,
              C=c_values,
              solver=solvers
            )

#Initialize StratifiedKfold for CV
cv = StratifiedKFold()

#Initialize GridSearchCV
gs = GridSearchCV(estimator=model,
                  param_grid=params,
                  cv=cv,
                  scoring='accuracy',
                  n_jobs=-1
                )

#Initialization done
print(gs)
gs.fit(X_train,y_train)

#Evalauting Best score,params
print("best_params_ : ", gs.best_params_)
print("best_score_ : ", gs.best_score_)

#making prediction
y_predgs = gs.predict(X_test)
y_predprobgs = gs.predict_proba(X_test)

#Evaluating metrics using GridSearchCV 
print("Accuracy Score is : ",accuracy_score(y_test,y_predgs))
print("Confusion Matrics : \n",confusion_matrix(y_test,y_predgs))
print("Classification Report :  \n",classification_report(y_test,y_predgs))

#USING RandomizedSearchCV
model = LogisticRegression()

#Initialize RandomizedSearchCV
rs = RandomizedSearchCV(estimator=model,
                        param_distributions=params,
                        scoring='accuracy',
                        cv=cv,
                        n_jobs=-1
                       )

print(rs)
rs.fit(X_train,y_train)

#Evalauting Best score,params
print("best_params_ : ", rs.best_params_)
print("best_score_ : ", rs.best_score_)


#making prediction
y_predrs = rs.predict(X_test)
y_predprobrs = rs.predict_proba(X_test)

#Evaluating metrics using GridSearchCV 
print("Accuracy Score is : ",accuracy_score(y_test,y_predrs))
print("Confusion Matrics : \n",confusion_matrix(y_test,y_predrs))
print("Classification Report :  \n",classification_report(y_test,y_predrs))