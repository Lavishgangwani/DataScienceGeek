import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


#load the dataset
data = load_diabetes()
df = pd.DataFrame(data=data.data , columns=data.feature_names)
df['target'] = data.target


#print(df.head())

#splitting the data
X = df.drop(columns=['target'] , axis=1)
y = df.target

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#checking correlation
sns.heatmap(X_train.corr(),annot=True)
plt.show()

#Intialize the model
dtr = DecisionTreeRegressor().fit(X_train,y_train)
y_pred = dtr.predict(X_test)

print("R2 Score : ",r2_score(y_test , y_pred))
print("MSE : ",mean_squared_error(y_test,y_pred))
print("MAE : ",mean_absolute_error(y_test,y_pred))

#plotting the graph
plot_tree(dtr,filled=True)
plt.show()

#   R2 Score :  -0.4837039995017023
#  MSE :  7608.3258426966295
#   MAE :  72.84269662921348

#Applying pre Pruning 
#hyperparmater Tuning
cri = ['squared_error','friedman_mse','absolute_error','poisson']
split = ['best' , 'random']
depth = [1,2,3,4,5,6,7,9,10]
max_feat = ['log2','sqrt','auto']

params = dict(criterion=cri,
              splitter = split,
              max_depth=depth,
              max_features=max_feat)

#Intializing GridSearchCV
gs = GridSearchCV(estimator=DecisionTreeRegressor(),
                  param_grid=params,
                  cv=5,
                  scoring='neg_mean_squared_error',
                  n_jobs=-1).fit(X_train,y_train)

y_predgs = gs.predict(X_test)

print("Best Score : ",gs.best_score_)
print("Best Params : ",gs.best_params_)

print("R2 Score : ",r2_score(y_test , y_predgs))
print("MSE : ",mean_squared_error(y_test,y_predgs))
print("MAE : ",mean_absolute_error(y_test,y_predgs))


#checking params of pre pruning
dtr2 = DecisionTreeRegressor(criterion='squared_error',
                             max_depth=2,
                             max_features='log2',
                             splitter='best').fit(X_train,y_train)

y_pred2 = dtr2.predict(X_test)

print("R2 Score : ",r2_score(y_test , y_pred2))
print("MSE : ",mean_squared_error(y_test,y_pred2))
print("MAE : ",mean_absolute_error(y_test,y_pred2))


#plotting the graph
plot_tree(dtr2,filled=True)
plt.show()




