#naive bayes.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB ,  BernoulliNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
import warnings
print(warnings.filterwarnings('ignore'))

#loading dataset
X,y = load_iris(return_X_y=True)

#splitting the data
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=22)

#Intialize the model
nb = GaussianNB().fit(X_train,y_train)
y_pred = nb.predict(X_test)

#print("Classification Report : \n",classification_report(y_test , y_pred))
#print("Confusion Matrix : \n",confusion_matrix(y_test , y_pred))

#loading another dataset
df = sns.load_dataset('tips')
print(df.dtypes)

cat_cols = df.select_dtypes(include='category')
for i in cat_cols.columns:
    print(f"Value counts for column {i}:")
    print(cat_cols[i].value_counts())

#Since day has 4 categories Traget variable should be day

#scalling the numerical cols
st = StandardScaler()
tips_scaled = df.copy() #copy of original dataframe
tips_scaled[['total_bill','size','tip']] = st.fit_transform(tips_scaled[['total_bill','size','tip']])

## One-hot encode categorical columns, excluding the target variable 'day'
tips_scaled = pd.get_dummies(data=tips_scaled , columns=['sex' , 'time', 'smoker'])

# Encode the target variable 'day' using LabelEncoder
le = LabelEncoder()
tips_scaled['day'] = le.fit_transform(tips_scaled['day'])

#print(df.head())
print(tips_scaled.head())


#Splitting the data
X_tips = tips_scaled.drop(columns=['day'] , axis=1)
y_tips = tips_scaled['day']


X_train_tips , X_test_tips , y_train_tips , y_test_tips = train_test_split(X_tips,
                                                                           y_tips,
                                                                           test_size=0.2,
                                                                           random_state=0)


#initialize the model
nb_tips = GaussianNB().fit(X_train_tips,y_train_tips)
y_pred_tips = nb_tips.predict(X_test_tips)

print("Classification Report : \n",classification_report(y_test_tips , y_pred_tips))
print("Confusion Matrix : \n",confusion_matrix(y_test_tips , y_pred_tips))


#Using BernoulliNB
nb_tips2 =BernoulliNB().fit(X_train_tips,y_train_tips)
y_pred_tips2 = nb_tips2.predict(X_test_tips)

print("Classification Report : \n",classification_report(y_test_tips , y_pred_tips2))
print("Confusion Matrix : \n",confusion_matrix(y_test_tips , y_pred_tips2))
