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
y_pred_proba = classifier.predict_proba(X_test)[:,1]

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print ROC AUC Score
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

#potting the graph
fpr , tpr , _ = roc_curve(y_test,y_pred_proba)
plt.figure()
plt.plot(fpr,tpr,color='darkorange',lw=2,label='ROC CURVE (area = %0.2f)' % roc_auc_score(y_test,y_pred_proba))
plt.plot([0,1] , [0,1] , color='navy' , lw=2 , linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#plot the decision Boundary
def plot_decision_boundary(X,y,model):
    plot_step=0.2
    x_min , x_max = X[:,0].min() - 1,X[:,0].max() + 1
    y_min , y_max = X[:,1].min() - 1,X[:,1].max() + 1
    xx , yy = np.meshgrid(
        np.arange(x_min , x_max , plot_step),
        np.arange(y_min , y_max , plot_step)
    )
    Z = model.predict(np.c_[xx.ravel() , yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=20)
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(X,y,classifier)

#Classification Report:
#              precision    recall  f1-score   support
#
#           0       1.00      0.29      0.44         7
#           1       0.96      1.00      0.98       113

#    accuracy                           0.96       120
#   macro avg       0.98      0.64      0.71       120
#weighted avg       0.96      0.96      0.95       120

#ROC AUC Score: 0.9557522123893806
#normal imbalance dataset report low Recall






#Now 1. RANDOM UNDERSAMPLING

from imblearn.under_sampling import RandomUnderSampler
#splitting the dataset
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Initialize RandomUnderSampler
rus = RandomUnderSampler()
X_resampled , y_resampled = rus.fit_resample(X_train,y_train)


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

#plotting the graph RandomUnderSampler
plt.figure(figsize=(8, 6))
plt.scatter(X_resampled[y_resampled == 0][:, 0], X_resampled[y_resampled == 0][:, 1], color='red', label='Class 1')
plt.scatter(X_resampled[y_resampled == 1][:, 0], X_resampled[y_resampled == 1][:, 1], color='blue', label='Class 2')
plt.title('2D Imbalanced Dataset with Two Classes (RandomUnderSampler)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

#Initialize the model
classifier_rus = LogisticRegression()
classifier_rus.fit(X_resampled,y_resampled)

#make predictions
y_pred = classifier_rus.predict(X_test)
y_pred_proba = classifier_rus.predict_proba(X_test)[:,1]

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print ROC AUC Score
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

#potting the graph
fpr , tpr , _ = roc_curve(y_test,y_pred_proba)
plt.figure()
plt.plot(fpr,tpr,color='darkorange',lw=2,label='ROC CURVE (area = %0.2f)' % roc_auc_score(y_test,y_pred_proba))
plt.plot([0,1] , [0,1] , color='navy' , lw=2 , linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

def plot_decision_boundary_rus(X,y,model):
    plot_step=0.2
    x_min , x_max = X[:,0].min() - 1,X[:,0].max() + 1
    y_min , y_max = X[:,1].min() - 1,X[:,1].max() + 1
    xx , yy = np.meshgrid(
        np.arange(x_min , x_max , plot_step),
        np.arange(y_min , y_max , plot_step)
    )
    Z = model.predict(np.c_[xx.ravel() , yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=20)
    plt.title("Decision Boundary (RandomUnderSampler)")
    plt.show()

plot_decision_boundary_rus(X,y,classifier_rus)
#Classification Report:
#              precision    recall  f1-score   support
#
#           0       0.13      1.00      0.23         7
#           1       1.00      0.59      0.74       113
#
#    accuracy                           0.62       120
#   macro avg       0.57      0.80      0.49       120
#weighted avg       0.95      0.62      0.71       120

#ROC AUC Score: 0.9469026548672567



#Now 2. RandomOverSampler

from imblearn.over_sampling import RandomOverSampler
#splitting the dataset
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Initialize RandomUnderSampler
ros = RandomOverSampler()
X_resampled_ros , y_resampled_ros = ros.fit_resample(X_train,y_train)


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

#plotting the graph RandomUnderSampler
plt.figure(figsize=(8, 6))
plt.scatter(X_resampled_ros[y_resampled_ros == 0][:, 0], X_resampled_ros[y_resampled_ros == 0][:, 1], color='red', label='Class 1')
plt.scatter(X_resampled_ros[y_resampled_ros == 1][:, 0], X_resampled_ros[y_resampled_ros == 1][:, 1], color='blue', label='Class 2')
plt.title('2D Imbalanced Dataset with Two Classes (RandomUnderSampler)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

#Initialize the model
classifier_ros = LogisticRegression()
classifier_ros.fit(X_resampled_ros,y_resampled_ros)

#make predictions
y_pred = classifier_ros.predict(X_test)
y_pred_proba = classifier_ros.predict_proba(X_test)[:,1]

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print ROC AUC Score
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

#potting the graph
fpr , tpr , _ = roc_curve(y_test,y_pred_proba)
plt.figure()
plt.plot(fpr,tpr,color='darkorange',lw=2,label='ROC CURVE (area = %0.2f)' % roc_auc_score(y_test,y_pred_proba))
plt.plot([0,1] , [0,1] , color='navy' , lw=2 , linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

def plot_decision_boundary_rs(X,y,model):
    plot_step=0.2
    x_min , x_max = X[:,0].min() - 1,X[:,0].max() + 1
    y_min , y_max = X[:,1].min() - 1,X[:,1].max() + 1
    xx , yy = np.meshgrid(
        np.arange(x_min , x_max , plot_step),
        np.arange(y_min , y_max , plot_step)
    )
    Z = model.predict(np.c_[xx.ravel() , yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=20)
    plt.title("Decision Boundary (RandomOverSampler)")
    plt.show()

plot_decision_boundary_rs(X,y,classifier_ros)
#Classification Report:
#             precision    recall  f1-score   support
#
#           0       0.18      1.00      0.31         7
#           1       1.00      0.73      0.84       113
#
#    accuracy                           0.74       120
#   macro avg       0.59      0.86      0.58       120
#weighted avg       0.95      0.74      0.81       120

#ROC AUC Score: 0.95448798988622





#Now 3. SMOTE

from imblearn.over_sampling import SMOTE

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying SMOTE
smote = SMOTE(random_state=42)
X_resampled_smote, y_resampled_smote = smote.fit_resample(X_train, y_train)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Class 1')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='Class 2')
plt.title('2D Imbalanced Dataset with Two Classes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X_resampled_smote[y_resampled_smote == 0][:, 0], X_resampled_smote[y_resampled_smote == 0][:, 1], color='red', label='Class 1')
plt.scatter(X_resampled_smote[y_resampled_smote == 1][:, 0], X_resampled_smote[y_resampled_smote == 1][:, 1], color='blue', label='Class 2')
plt.title('2D Imbalanced Dataset with Two Classes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()


# Initialize and train RandomForest classifier on resampled data
classifier_smote = LogisticRegression()
classifier_smote.fit(X_resampled_smote, y_resampled_smote)

# Predict test set
y_pred_smote = classifier_smote.predict(X_test)
y_proba_smote = classifier_smote.predict_proba(X_test)[:, 1]

# Print classification report for SMOTE data
print("Classification Report (With SMOTE):")
print(classification_report(y_test, y_pred_smote))

# Print ROC AUC Score for SMOTE data
print("ROC AUC Score (With SMOTE):", roc_auc_score(y_test, y_proba_smote))

# Plotting ROC AUC Curve for SMOTE data
fpr_smote, tpr_smote, _ = roc_curve(y_test, y_proba_smote)
plt.figure()
plt.plot(fpr_smote, tpr_smote, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_proba_smote))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (With SMOTE)')
plt.legend(loc="lower right")
plt.show()

# Function to plot decision boundaries for SMOTE data
def plot_decision_boundaries_smote(X, y, model):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=20)
    plt.title("Decision Boundary (With SMOTE)")
    plt.show()

# Plot decision boundary for SMOTE data
plot_decision_boundaries_smote(X, y, classifier_smote)
#Classification Report (With SMOTE):
#              precision    recall  f1-score   support

#           0       0.17      1.00      0.30         4
#           1       1.00      0.75      0.86        76

#    accuracy                           0.76        80
#   macro avg       0.59      0.88      0.58        80
#weighted avg       0.96      0.76      0.83        80
#ROC AUC Score (With SMOTE): 0.950657894736842





#now 4. Ensemble Method (Balanced RF)
from imblearn.ensemble import BalancedRandomForestClassifier

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying SMOTE
classifier_BRF = BalancedRandomForestClassifier(random_state=42)
classifier_BRF.fit(X_train, y_train)

# Predict test set
y_pred_brf = classifier_BRF.predict(X_test)
y_proba_brf = classifier_BRF.predict_proba(X_test)[:, 1]

# Print classification report for SMOTE data
print("Classification Report (With SMOTE):")
print(classification_report(y_test, y_pred_brf))

# Print ROC AUC Score for SMOTE data
print("ROC AUC Score (With SMOTE):", roc_auc_score(y_test, y_proba_brf))

# Plotting ROC AUC Curve for SMOTE data
fpr_brf, tpr_brf, _ = roc_curve(y_test, y_proba_brf)
plt.figure()
plt.plot(fpr_brf, tpr_brf, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_proba_brf))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (With SMOTE)')
plt.legend(loc="lower right")
plt.show()

# Function to plot decision boundaries for SMOTE data
def plot_decision_boundaries_brf(X, y, model):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=20)
    plt.title("Decision Boundary (With BRF)")
    plt.show()

# Plot decision boundary for BRF data
plot_decision_boundaries_brf(X, y, classifier_BRF)
#Classification Report (With BRF):
#              precision    recall  f1-score   support

#           0       0.16      0.75      0.26         4
#           1       0.98      0.79      0.88        76

#    accuracy                           0.79        80
#   macro avg       0.57      0.77      0.57        80
#weighted avg       0.94      0.79      0.85        80

#ROC AUC Score (With SMOTE): 0.8519736842105262






#Now 5. Custom Sensitive Learning
#Part-1 Class Weights

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a logistic regression model with class weights
model = LogisticRegression(class_weight={0:50,1:1}, solver='liblinear')

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
y_proba = classifier.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Print ROC AUC Score
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Plotting ROC AUC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_proba))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Function to plot decision boundaries
def plot_decision_boundaries_cw(X, y, model):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=20)
    plt.title("Decision Boundary Class Weights")
    plt.show()

# Plot decision boundary
plot_decision_boundaries_cw(X, y, model)
#Classification Report:
#              precision    recall  f1-score   support
#
#           0       0.09      1.00      0.16         7
#           1       1.00      0.37      0.54       113
#
#    accuracy                           0.41       120
#   macro avg       0.54      0.69      0.35       120
#weighted avg       0.95      0.41      0.52       120

#ROC AUC Score: 0.9557522123893806





#Part2 : Custom Loss Function
import xgboost as xgb
# Generate imbalanced dataset
n_samples_1 = 25  # Number of samples in class 1
n_samples_2 = 375  # Number of samples in class 2
centers = [(0, 0), (2, 2)]  # Centers of each cluster
cluster_std = [1.5, 1.5]  # Standard deviation of each cluster

X, y = make_blobs(n_samples=[n_samples_1, n_samples_2],
                  centers=centers,
                  cluster_std=cluster_std,
                  random_state=0)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def custom_loss(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))  # Convert to probability

    # Define penalties
    false_positive_penalty = 10
    false_negative_penalty = 1.0

    grad = (preds - labels) * ((labels == 1) * false_negative_penalty + (labels == 0) * false_positive_penalty)
    hess = preds * (1 - preds) * ((labels == 1) * false_negative_penalty + (labels == 0) * false_positive_penalty)
    return grad, hess

# Convert to DMatrix object
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set up parameters
params = {
    'max_depth': 3,
    'eta': 0.1,
    'silent': 1,
}

# Train the model
bst = xgb.train(params, dtrain, num_boost_round=10, obj=custom_loss)

# Predict test set
y_pred = np.where(bst.predict(dtest) > 0.5, 1, 0)
y_proba = bst.predict(dtest)  # Probability predictions

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print ROC AUC Score
auc_score = roc_auc_score(y_test, y_proba)
print("ROC AUC Score:", auc_score)

# Plotting ROC AUC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

def plot_decision_boundaries_clf(X, y, model):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = model.predict(xgb.DMatrix(np.c_[xx.ravel(), yy.ravel()]))
    Z = np.where(Z > 0.5, 1, 0).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=20)
    plt.title("Decision Boundary")
    plt.show()

# Plot decision boundary
plot_decision_boundaries_clf(X, y, bst)
#Classification Report:
#              precision    recall  f1-score   support
#
#           0       0.17      0.57      0.27         7
#           1       0.97      0.83      0.90       113
#
#    accuracy                           0.82       120
#   macro avg       0.57      0.70      0.58       120
#weighted avg       0.92      0.82      0.86       120

#ROC AUC Score: 0.7319848293299621