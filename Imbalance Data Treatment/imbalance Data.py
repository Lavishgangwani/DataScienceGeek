#imbalance.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report , roc_auc_score , roc_curve , confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

#Generate imbalance dataset
np.random.seed(42)

n_samples1 = 25
n_samples2 = 375
centers = [(0,0) , (2,2)]
clusters = [1.5,1.5]

#creating a dataset
X , y = make_blobs(n_samples=[n_samples1,n_samples2],
                   centers=centers,
                   cluster_std=clusters,
                   random_state=0)


#plotting the graph
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 1')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 2')
plt.title('2D Imbalanced Dataset with Two Classes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

#splitting the dataset
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Initialize the model
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

#make predictions
y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print ROC AUC Score
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

#potting the graph



