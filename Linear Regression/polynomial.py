#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

#creating a dataset
X = 6 * np.random.rand(100,1) - 3
y = 0.5 * X**2 + 1.5 * X + 2 + np.random.randn(100,1)

#adding some outlier with the data 
#poly equation is : ax^2 + bx + 2 where a = 0.5 and b = 1.5

#plotting the graph 
plt.scatter(X, y, color='g')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Scatter Plot of X vs y')
plt.show() 

#splitting the data
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=1)

#applying first Linear regression
lr = LinearRegression().fit(X_train,y_train)

#metrics evaluation
print("R2 FOR LinearRegression : ",r2_score(y_test,lr.predict(X_test)))

