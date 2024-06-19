#svr.py


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split , GridSearchCV , RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler , OneHotEncoder , LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error


#loading dataset 
df = sns.load_dataset('tips')
print(df.head())
print(df.columns)

#Splitting the data
X = df[['tip', 'sex', 'smoker', 'day', 'time', 'size']]
y = df['total_bill']

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=21)
#print(X_train)


#encoding categorical cols
le1=LabelEncoder()
le2=LabelEncoder()
le3=LabelEncoder()

X_train['sex'] = le1.fit_transform(X_train['sex'])
X_train['smoker'] = le2.fit_transform(X_train['smoker'])
X_train['time'] = le3.fit_transform(X_train['time'])

X_test['sex'] = le1.transform(X_test['sex'])
X_test['smoker'] = le2.transform(X_test['smoker'])
X_test['time'] = le3.transform(X_test['time'])

#Now Encoding time col using OneHotEncoder through column transformer
ct = ColumnTransformer(transformers=[('OHE',
                                     OneHotEncoder(drop='first'),
                                     [3])],
                                     remainder='passthrough')


print(ct)
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

#Intialize SVR
svr = SVR()
svr.fit(X_train,y_train)

y_pred = svr.predict(X_test)
print("R2 Score : ", r2_score(y_test,y_pred))
print('MAE : ', mean_absolute_error(y_test,y_pred))
print('MSE : ',mean_squared_error(y_test , y_pred))


#HYPERPARAMETER TUNING
c_values = [0.1,0.01,1.0,10,100,1000]
gammas =  [0.1,0.01,0.001,0.0001,1.0,10]
kernels = ['rbf']

params = dict(C=c_values,
              gamma=gammas,
              kernel=kernels)

#Initilaize GridSearchCV
gs = GridSearchCV(estimator=SVR() , 
                  param_grid=params,
                  scoring='r2',
                  verbose=3,
                  refit=True,
                  n_jobs=-1,
                  cv=5)

gs.fit(X_train,y_train)
print("Best Params : ",gs.best_params_)
print("Best Score : ",gs.best_score_)
print('R2 Score GSearchCV: \n',r2_score(y_test,gs.predict(X_test)))
print("MSE GSearchCV : \n" , mean_squared_error(y_test ,gs.predict(X_test)))


#Hyperparameter Tuning Using RandomizedSearchCV
c_values = [0.1,0.01,1.0,10,100,1000]
gammas =  [0.1,0.01,0.001,0.0001,1.0,10]
kernels = ['rbf']

params = dict(C=c_values,
              gamma=gammas,
              kernel=kernels)

#Initilaize GridSearchCV
rs = RandomizedSearchCV(estimator=SVR() , 
                  param_distributions=params,
                  scoring='r2',
                  verbose=3,
                  refit=True,
                  n_jobs=-1,
                  cv=5)

rs.fit(X_train,y_train)
print("Best Params : ",rs.best_params_)
print("Best Score : ",rs.best_score_)
print('R2 Score RSearchCV: \n',r2_score(y_test,rs.predict(X_test)))
print("MSE RSearchCV : \n" , mean_squared_error(y_test ,rs.predict(X_test)))




