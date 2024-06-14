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
from collections import Counter



#creating a dataset
X,y = make_classification(n_samples=10000,
                          n_clusters_per_class=2,
                          n_features=2,
                          n_redundant=0,
                          weights=[.99],
                          random_state=21
                        )


#splitting the dataset
X_train , X_test ,y_train , y_test = train_test_split(X,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=2)


#imbalance class
print(Counter(y))

#plotting the graph of imbalance classes
sns.scatterplot(x=pd.DataFrame(X)[0],y=pd.DataFrame(X)[1],hue=y)
plt.show()


## Hyperparameter Tuning and Cross Validation

#Using GridSearchCV  
model = LogisticRegression()
penalty = ['l1','l2','elasticnet']
c_values = [100,10,1.0,0.1,0.01,0.001,0.0001]
solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
weights = [{0: w, 1: y} for w in [1,10,50,100] for y in [1,10,50,100]]

#checking weights parameter
#here for imbalance dataset treatment we have initialize the class_weight parameter in Logistic Regression
print(weights)


#Converting all the parameters into dict
params = dict(penalty=penalty,
              C=c_values,
              solver=solvers,
              class_weight=weights
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
