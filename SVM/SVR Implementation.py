#svr.py


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split , GridSearchCV , RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error


#loading dataset 
df = sns.load_dataset('tips')
print(df.head())
print(df.columns)

#Splitting the data
X = []


