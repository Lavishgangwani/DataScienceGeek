# polynomial.py

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def create_dataset():
    X = 6 * np.random.rand(100, 1) - 3
    y = 0.5 * X**2 + 1.5 * X + 2 + np.random.randn(100, 1)
    return X, y

def plot_data(X, y):
    plt.scatter(X, y, color='g')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Scatter Plot of X vs y')
    plt.show()

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=1)

def linear_regression(X_train, y_train, X_test, y_test):
    lr = LinearRegression().fit(X_train, y_train)
    print("R2 FOR LinearRegression: ", r2_score(y_test, lr.predict(X_test)))
    plt.scatter(X_train, y_train)
    plt.plot(X_train, lr.predict(X_train), color='r')
    plt.title('Linear Regression (Best fit line)', weight='bold')
    plt.show()
    return lr


def polynomial_regression(X_train, y_train, X_test, y_test, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    lr_poly = LinearRegression().fit(X_train_poly, y_train)
    y_pred_poly = lr_poly.predict(X_test_poly)
    print(f"R2 Score (Poly, degree={degree}): ", r2_score(y_test, y_pred_poly))
    plt.scatter(X_train, y_train)
    plt.scatter(X_train, lr_poly.predict(X_train_poly), color='r')
    plt.title('Polynomial Regression (Parabola Curve)', weight='bold')
    plt.show()
    return lr_poly, poly



#creation of prediction dataset & visualize
def pred_dataset(X_train, y_train, X_test, y_test, lr_poly, poly):
    X_new = np.linspace(-3,3,200).reshape(200,1)
    X_new_poly = poly.fit_transform(X_new)
    #print(X_new_poly)
    y_pred_new = lr_poly.predict(X_new_poly)
    plt.plot(X_new, y_pred_new, "r-", linewidth=2, label=" New Predictions")
    plt.plot(X_train, y_train, "b.",label='Training points')
    plt.plot(X_test, y_test, "g.",label='Testing points')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()



## Calling of the functions(By Encapsulation It'll link with pipeline.py)


# Generate and plot dataset
X, y = create_dataset()
plot_data(X, y)

# Split the data
X_train, X_test, y_train, y_test = split_data(X, y)

# Apply Linear Regression and visualize
linear_regression(X_train, y_train, X_test, y_test)

# Apply Polynomial Regression and capture the returned values
lr_poly, poly = polynomial_regression(X_train, y_train, X_test, y_test, degree=2)

# Visualize pred_new dataset
pred_dataset(X_train, y_train, X_test, y_test, lr_poly, poly)