#K-nearest Neighbor

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#generating a dataset
X,y = make_classification(n_classes=2,
                          n_samples=1000,
                          n_features=3,
                          n_redundant=1,
                          random_state=0)

#splitting the data
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Initialize the model
knn= KNeighborsClassifier(n_neighbors=5,algorithm='auto').fit(X_train,y_train)
y_pred = knn.predict(X_test)

print("accuracy_score : ",accuracy_score(y_test,y_pred))
print("Classification Report : \n",classification_report(y_test,y_pred))
print("confusion_matrix : \n",confusion_matrix(y_test,y_pred))


#HYPERPARAMETER TUNING
n = [k for k in range(1,10)]
weights= ['uniform' , 'distance']
algo = ['kd_tree','ball_tree','auto','brute']
p = [1,2]

params = dict(n_neighbors=n,
              weights=weights,
              algorithm=algo,
              p=p)

print("Params : \n",params)
#Initialize GridSearchCV
gs = GridSearchCV(estimator=KNeighborsClassifier(),
                param_grid=params,
                cv=5,
                scoring='accuracy',
                n_jobs=-1)

gs.fit(X_train,y_train)
y_predgs = gs.predict(X_test)

print("Best_score : ",gs.best_score_)
print("Best_Param : ",gs.best_params_)

print("accuracy_score : ",accuracy_score(y_test,y_predgs))
print("Classification Report : \n",classification_report(y_test,y_predgs))
print("confusion_matrix : \n",confusion_matrix(y_test,y_predgs))


#Checking GridSearchCV Params
knn1 = KNeighborsClassifier(n_neighbors=4,
                            p=1,
                            weights='distance',
                            algorithm='kd_tree')

knn1.fit(X_train,y_train)
y_predknn = knn1.predict(X_test)

print("accuracy_score : ",accuracy_score(y_test,y_predknn))
print("Classification Report : \n",classification_report(y_test,y_predknn))
print("confusion_matrix : \n",confusion_matrix(y_test,y_predknn))

