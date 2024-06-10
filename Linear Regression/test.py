from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np


data = load_breast_cancer()
df = pd.DataFrame(data.data , columns=data.feature_names)
df['target'] = data.target

print(df['target'].value_counts())