import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing(as_frame=True)
X = housing.data.values
y = housing.target.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) #to perform standardisation
# Split the train and test Data
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=1/3, random_state=0) #66,33
class LassoRegressionScratch:
    def __init__(self, learning_rate=0.01, iterations=1000, l1_lambda=0.01):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_lambda = l1_lambda
        self.w = None
        self.b = None

    def fit(self, X, y):
        self.m, self.n = X.shape #the number of rows and columns
        self.w = np.zeros(self.n)
        self.b = 0
        
        for i in range(self.iterations):
            self.update_weights(X, y)
        return self

    def update_weights(self, X, y):
        y_pred = self.predict(X)
        # Gradient for weights
        dw = np.zeros(self.n)
        for j in range(self.n):
            # L1 derivative: lambda * sign(w)
            if self.w[j] > 0:
                l1_deriv = self.l1_lambda
            elif self.w[j] < 0:
                l1_deriv = -self.l1_lambda
            else:
                l1_deriv = 0
            
            dw[j] = (-2 * (X[:, j].dot(y - y_pred)) + l1_deriv) / self.m
        db = -2 * np.sum(y - y_pred) / self.m
        # Update
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db
    def predict(self, X):
        return X.dot(self.w) + self.b
model = LassoRegressionScratch(iterations=1000, learning_rate=0.01, l1_lambda=0.1)
model.fit(x_train, y_train)#training
y_pred = model.predict(x_test)#evaluation
print("Predicted values: ", y_pred[:3])
print("Real values:      ", y_test[:3])
print("Trained b:        ", model.b)
class RidgeRegressionScratch:
    def __init__(self, learning_rate=0.01, iterations=1000, l2_lambda=1.0):
        self.lr = learning_rate
        self.iterations = iterations
        self.l2_lambda = l2_lambda
        self.w = None
        self.b = None

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        
        for _ in range(self.iterations):
            y_pred = self.predict(X)
            
            # Gradient calculation
            # Gradient of Mean Squared Error + Gradient of (lambda * w^2)
            dw = (-(2 * X.T.dot(y - y_pred)) + (2 * self.l2_lambda * self.w)) / self.m#Y_pred is the cost function

            db = -2 * np.sum(y - y_pred) / self.m
            
            # Update weights
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def predict(self, X):
        return X.dot(self.w) + self.b
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=1/3, random_state=0) 
model = RidgeRegressionScratch(learning_rate=0.01, iterations=1000, l2_lambda=0.5)
model.fit(X_train, Y_train)#training
Y_pred = model.predict(X_test)#prediction
print("Predicted values: ",Y_pred[:3])
print("Real values:      ",Y_test[:3])
print("Trained b:        ",model.b)
#Visualizing
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='violet')
plt.plot([y_train.min(), y_train.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs predicted using Lasso')
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.grid(True, linestyle='--', alpha=0.6)
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred, alpha=0.5, color='blue')
plt.plot([Y_train.min(), Y_train.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Housing Prices Ridge')


plt.show()
print(x_train[:,2])
