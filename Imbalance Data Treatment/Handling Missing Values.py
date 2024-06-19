#Handling Missing Values.py

#import libraries
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns


#loading dataset

data = sns.load_dataset('titanic')
#print(data)

#check any missing values
missing_values = data.isnull().sum()
#print(missing_values)

#NOW There are generally three types of missing values:

#1. **Missing Completely at Random (MCAR):**
#In this type, the missingness of data is completely random and unrelated to any other variables, observed or unobserved. There's no systematic reason for the missing values. An example of MCAR might be a weather station failing to record temperature readings due to random equipment failure.

#2. **Missing at Random (MAR):**
#   MAR occurs when the missingness is related to observed data but not to the missing data itself. In other words, the propensity for a data point to be missing is related to some other observed variable(s). For instance, in a survey, if women are less likely to report their income than men, but income is not related to gender itself, then the missingness of income data is MAR.

#3. **Missing Not at Random (MNAR):**
#   MNAR occurs when the missingness is related to the missing data itself, even after controlling for observed data. In this case, the missingness is related to the value of the missing data. An example of MNAR could be if high-income individuals are less likely to report their income levels in a survey.

#Let's consider an example dataset where we're collecting information about students, including their gender, age, and test scores. Suppose some students didn't provide their test scores:

#| Student ID | Gender | Age | Test Score |
#|------------|--------|-----|------------|
#| 1          | Male   | 18  | 85         |
#| 2          | Female | 17  | 75         |
#| 3          | Male   | 16  | NaN        |
#| 4          | Female | 18  | 92         |
#| 5          | Male   | 17  | NaN        |

#In this example:
#- If the missing test scores are due to random technical issues during data collection, it's MCAR.
#- If, say, boys are more likely to skip the test, but the reason is not because of their gender but perhaps their interest in the subject, it's MAR.
#- If the reason for missing test scores is directly related to their performance (e.g., students with low scores are more likely to skip), it's MNAR.

#Understanding the type of missingness is crucial because it can impact how you handle missing data and the validity of your analyses.

#imputation Techniques are

# 1. MEAN IMPUTATION
data['age_mean'] = data['age'].fillna(data['age'].mean())
#print(data[['age_mean', 'age']])

# 1. MEDIAN IMPUTATION
data['age_median'] = data['age'].fillna(data['age'].median())
#print(data[['age_median','age_mean','age']])


## 3. SIMPLE IMPUTER(SKlearn lib)
from sklearn.impute import SimpleImputer
# Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
data[['age_imputed']] = imputer.fit_transform(data[['age']])

# Print the original 'age' column, along with imputed columns for comparison
print(data[['age', 'age_imputed']])