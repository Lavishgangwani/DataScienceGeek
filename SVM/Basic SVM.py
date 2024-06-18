#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split , GridSearchCV , RandomizedSearchCV
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.svm import SVC



#creating a dataset
X , y = make_classification(n_samples=1000,
                            n_classes=2,
                            n_clusters_per_class=2,
                            n_informative=2,
                            n_features=2,
                            n_redundant=0,
                            random_state=23)


#visualaize data
sns.scatterplot(x=pd.DataFrame(X)[0] , y=pd.DataFrame(X)[1] , hue=y)
plt.show()


#splitting the data
X_train , X_test, y_train , y_test = train_test_split(X,y,test_size=0.3,random_state=28)


#Initialize SVM (kernel='Linear')
svm = SVC(kernel='linear').fit(X_train,y_train)
y_pred = svm.predict(X_test)

#"coef_ is only available when using a linear kernel"
print("Coefficients : ",svm.coef_)

print('Confusion Metrics : \n',confusion_matrix(y_test,y_pred))
print("Classification Report : \n" , classification_report(y_test , y_pred))



#Initialize SVM (kernel='rbf')
svm_rbf = SVC(kernel='rbf').fit(X_train,y_train)
y_pred1 = svm_rbf.predict(X_test)
print('Confusion Metrics RBF : \n',confusion_matrix(y_test,y_pred1))
print("Classification Report RBF : \n" , classification_report(y_test , y_pred1))


#Initialize SVM (kernel='poly')
svm_poly = SVC(kernel='poly').fit(X_train,y_train)
y_pred2 = svm_poly.predict(X_test)
print('Confusion Metrics POLY : \n',confusion_matrix(y_test,y_pred2))
print("Classification Report POLY : \n" , classification_report(y_test , y_pred2))


#Initialize SVM (kernel='sigmoid')
svm_sg = SVC(kernel='sigmoid').fit(X_train,y_train)
y_pred3 = svm_sg.predict(X_test)
print('Confusion Metrics SIGMOID: \n',confusion_matrix(y_test,y_pred3))
print("Classification Report SIGMOID : \n" , classification_report(y_test , y_pred3))


#HYPERPARAMETER TUNING
c_values = [0.1,0.01,1.0,10,100,1000]
gammas =  [0.1,0.01,0.001,0.0001,1.0,10]
kernels = ['linear' ,'rbf' , 'poly' , 'sigmoid']

params = dict(C=c_values,
              gamma=gammas,
              kernel=kernels)

#Initilaize GridSearchCV
gs = GridSearchCV(estimator=SVC() , 
                  param_grid=params,
                  scoring='accuracy',
                  verbose=3,
                  refit=True,
                  n_jobs=-1,
                  cv=5)

gs.fit(X_train,y_train)
print("Best Params : ",gs.best_params_)
print("Best Score : ",gs.best_score_)
print('Confusion Metrics GSearchCV: \n',confusion_matrix(y_test,gs.predict(X_test)))
print("Classification Report GSearchCV : \n" , classification_report(y_test ,gs.predict(X_test)))




