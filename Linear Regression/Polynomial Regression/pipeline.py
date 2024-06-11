# pipeline.py

from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from polynomial import create_dataset, split_data, polynomial_regression

# Create dataset
X, y = create_dataset()

# Split the data
X_train, X_test, y_train, y_test = split_data(X, y)

# Function to perform polynomial regression with pipeline
def poly_regression(degree):
    X_new = np.linspace(-3, 3, 200).reshape(200, 1)
    
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    lin_reg = LinearRegression()
    poly_regression = Pipeline([
        ("poly_features", poly_features),
        ("lin_reg", lin_reg)
    ])
    
    poly_regression.fit(X_train, y_train)  # Polynomial and fit of linear regression
    y_pred_new = poly_regression.predict(X_new)
    
    # Plotting prediction line
    plt.plot(X_new, y_pred_new, 'r', label="Degree " + str(degree), linewidth=2)
    plt.plot(X_train, y_train, "b.", linewidth=3)
    plt.plot(X_test, y_test, "g.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.axis([-4, 4, 0, 10])
    plt.show()

# Run polynomial regression with the desired degree
poly_regression(degree=2)
