#Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#creating a dataset
X,y = make_classification(n_samples=1000,
                          n_informative=2,
                          n_features=2,
                          n_classes=2,
                          n_redundant=0,
                          random_state=43)


#splitting the data
X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=2)


#creating a Dummy Model first with default 0 as output 
dummy_model_prob = [0 for _ in range(len(y_test))]
print(dummy_model_prob)

#Initialize Logistic Regression
lr = LogisticRegression().fit(X_train,y_train)
model_prob = lr.predict_proba(X_test)

#print(model_prob)

#lets focus on positive outcome
model_prob = model_prob[:,1]

#Calcluate the score
dummy = roc_auc_score(y_test,dummy_model_prob)
original_lr = roc_auc_score(y_test,model_prob)
print("ROC AUC SCORE : ",dummy)
print("ROC AUC SCORE : ",original_lr)

#OUTPUTS :
##ROC AUC SCORE :  0.5
##ROC AUC SCORE :  0.9859836102279342

#It Means that Bydefault Internally In scikit Learn threshold for any predict outcome is 0.5 as per dummy model prob

#![image.png](attachment:image.png)
#![image-2.png](attachment:image-2.png)

#calculate roc_curve 
dummy_fpr , dummy_tpr , _ = roc_curve(y_test,dummy_model_prob)
model_fpr , model_tpr , threshold = roc_curve(y_test,model_prob)

#plotting the graph
plt.plot(dummy_fpr,dummy_tpr,linestyle='--', label='Dummy Model')
plt.plot(model_fpr,model_tpr,marker='.' , label='Logistic')
plt.xlabel('FALSE POSITIVE RATE')
plt.ylabel('TRUE POSITIVE RATE')
plt.legend()
plt.show()


#Now plotting graph for threshold visualizations very closely
fig = plt.figure(figsize=(20,50))
plt.plot(dummy_fpr,dummy_tpr , linestyle='--',label='Dummy Model')
plt.plot(model_fpr , model_tpr , marker='.',label='Logistic')
plt.xlabel('FALSE POSITIVE RATE')
plt.ylabel('TRUE POSITIVE RATE')
ax = fig.add_subplot(111)
for xyz in zip(model_fpr,model_tpr,threshold):
    ax.annotate('%s' % np.round(xyz[2],2), xy=(xyz[0],xyz[1]))

plt.xlabel('FALSE POSITIVE RATE')
plt.ylabel('TRUE POSITIVE RATE')
plt.legend()
plt.show()
