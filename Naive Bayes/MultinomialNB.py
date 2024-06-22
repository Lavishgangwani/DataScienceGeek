import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix

# Load the tips dataset from seaborn
df = sns.load_dataset('tips')

# Scaling the numerical columns using MinMaxScaler
mm_scaler = MinMaxScaler()
tips_scaled = df.copy()  # Make a copy of the original DataFrame
tips_scaled[['total_bill', 'size', 'tip']] = mm_scaler.fit_transform(tips_scaled[['total_bill', 'size', 'tip']])

# One-hot encode categorical columns, excluding the target variable 'day'
tips_scaled = pd.get_dummies(tips_scaled, columns=['sex', 'smoker', 'time'])

# Encode the target variable 'day' using LabelEncoder
le = LabelEncoder()
tips_scaled['day'] = le.fit_transform(tips_scaled['day'])

# Separate features and target variable
X = tips_scaled.drop(columns=['day'])
y = tips_scaled['day']

# Split the data into training and testing sets
X_train_tips, X_test_tips, y_train_tips, y_test_tips = train_test_split(X, y, test_size=0.2, random_state=42)

# Using MultinomialNB
nb_tips1 = MultinomialNB().fit(X_train_tips, y_train_tips)
y_pred_tips1 = nb_tips1.predict(X_test_tips)

print("MultinomialNB Classification Report : \n", classification_report(y_test_tips, y_pred_tips1))
print("MultinomialNB Confusion Matrix : \n", confusion_matrix(y_test_tips, y_pred_tips1))

# Using BernoulliNB
nb_tips2 = BernoulliNB().fit(X_train_tips, y_train_tips)
y_pred_tips2 = nb_tips2.predict(X_test_tips)

print("BernoulliNB Classification Report : \n", classification_report(y_test_tips, y_pred_tips2))
print("BernoulliNB Confusion Matrix : \n", confusion_matrix(y_test_tips, y_pred_tips2))


#The error you're encountering indicates that the MultinomialNB model cannot handle 
# negative values in the input data. MultinomialNB is designed for discrete data, 
# such as word counts in text classification, 
# and it expects all feature values to be non-negative. 
# However, your numerical features are scaled using StandardScaler, which can produce negative values.

#To address this, you can use MinMaxScaler instead of StandardScaler to scale your numerical features.
#MinMaxScaler scales the data to a range between 0 and 1, which is suitable for MultinomialNB. 
# Additionally, ensure that your one-hot encoded categorical variables are not altered during the scaling process.