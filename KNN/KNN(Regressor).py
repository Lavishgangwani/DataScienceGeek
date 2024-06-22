#K-nearest Neighbor

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor , KDTree
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import warnings
print(warnings.filterwarnings('ignore'))

#generating a dataset
X,y = make_regression(n_features=3,
                      n_samples=1000,
                      noise=10,
                      random_state=0)
#splitting the data
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Initialize the model
knn= KNeighborsRegressor(n_neighbors=5,algorithm='auto').fit(X_train,y_train)
y_pred = knn.predict(X_test)

print("r2_score : ",r2_score(y_test,y_pred))
print("mean_squared_error : ",mean_squared_error(y_test,y_pred))
print("mean_absolute_error : ",mean_absolute_error(y_test,y_pred))

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
gs = GridSearchCV(estimator=KNeighborsRegressor(),
                param_grid=params,
                cv=5,
                scoring='accuracy',
                n_jobs=-1)

gs.fit(X_train,y_train)
y_predgs = gs.predict(X_test)

print("Best_score : ",gs.best_score_)
print("Best_Param : ",gs.best_params_)

print("r2_score : ",r2_score(y_test,y_predgs))
print("mean_squared_error : ",mean_squared_error(y_test,y_predgs))
print("mean_absolute_error : ",mean_absolute_error(y_test,y_predgs))


#checking params
knn1= KNeighborsRegressor(n_neighbors=1,
                          algorithm='kd_tree',
                          p=1,
                          weights='uniform').fit(X_train,y_train)
y_pred1 = knn1.predict(X_test)

print("r2_score : ",r2_score(y_test,y_pred1))
print("mean_squared_error : ",mean_squared_error(y_test,y_pred1))
print("mean_absolute_error : ",mean_absolute_error(y_test,y_pred1))

