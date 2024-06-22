import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


#loading dataset
data = load_iris()
df = pd.DataFrame(data=data.data , columns=data.feature_names)
df['target'] = data.target

print(data.DESCR)


#splitting the data
X = df.drop(columns=['target'] , axis=1)
y = df.target

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Initialize DT
dt = DecisionTreeClassifier().fit(X_train,y_train)
y_pred = dt.predict(X_test)

print("accuracy Score : ",accuracy_score(y_test , y_pred))
print("Classification Report : \n",classification_report(y_test , y_pred))
print("Confusion Matrix : \n",confusion_matrix(y_test , y_pred))

#plotting the tree
plot_tree(dt , filled=True)
plt.show()

##Pre-Pruning (Early Stopping)

#In pre-pruning, the tree building process is halted early based on certain criteria. These criteria are specified as hyperparameters before the tree is constructed. Common pre-pruning parameters include:

    #Max Depth: Limits the maximum depth of the tree. If the depth exceeds this value, further splits are stopped.
    #Min Samples Split: The minimum number of samples required to split a node. If a node has fewer than this number of samples, it will not be split further.
    #Min Samples Leaf: The minimum number of samples required to be in a leaf node. If a split results in a child node with fewer than this number of samples, the split is discarded.
    #Max Features: The maximum number of features to consider when looking for the best split. This parameter is useful in ensemble methods like Random Forests.


#Post-Pruning (Late Pruning)

#In post-pruning, the tree is fully grown without any early stopping. Once the tree is completely built, it is then pruned by removing branches that have little importance or do not improve the model's performance significantly. Common post-pruning techniques include:

    #Cost Complexity Pruning: This involves evaluating the tree with different levels of pruning and selecting the level that optimizes a trade-off between the tree's complexity and its performance on validation data.
    #Reduced Error Pruning: Nodes are removed if the tree's performance on a validation set does not decrease after the removal.

#APPLYING POST PRUNING
#Initialize DT
dt1 = DecisionTreeClassifier(max_depth=2).fit(X_train,y_train)
y_pred1 = dt1.predict(X_test)

print("accuracy Score : ",accuracy_score(y_test , y_pred1))
print("Classification Report : \n",classification_report(y_test , y_pred1))
print("Confusion Matrix : \n",confusion_matrix(y_test , y_pred1))

#plotting the tree
plot_tree(dt1 , filled=True)
plt.show()


#APPLYING PRE PRUNING
cri = ['gini','entropy','log_loss']
split = ['best' , 'random']
depth = [1,2,3,4,5,6,7,8]
max_feat = ['auto' , 'log2','sqrt']

params = dict(criterion=cri,
              splitter=split,
              max_depth=depth,
              max_features=max_feat)

print("Params : ",params)

#Initialize GridSearchCV
gs = GridSearchCV(estimator=DecisionTreeClassifier(),
                  param_grid=params,
                  cv=3,
                  n_jobs=-1,
                  scoring='accuracy')

gs.fit(X_train,y_train)
y_predgs = gs.predict(X_test)

print('Best Score' , gs.best_score_)
print('Best params : ',gs.best_params_)

print("accuracy Score : ",accuracy_score(y_test , y_predgs))
print("Classification Report : \n",classification_report(y_test , y_predgs))
print("Confusion Matrix : \n",confusion_matrix(y_test , y_predgs))


#After pre pruning checking params
dt2 = DecisionTreeClassifier(criterion='log_loss',
                             max_depth=4,
                             max_features='sqrt',
                             splitter='random').fit(X_train,y_train)

y_pred2 = dt2.predict(X_test)
print("accuracy Score : ",accuracy_score(y_test , y_pred2))
print("Classification Report : \n",classification_report(y_test , y_pred2))
print("Confusion Matrix : \n",confusion_matrix(y_test , y_pred2))


#plotting graph
plot_tree(dt2,filled=True)
plt.show()




