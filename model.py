# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

# Importing the dataset
data = pd.read_csv('./data/Salary_Data.csv')

# Isolate your data columns
salary = data.iloc[:,0].values
years_exp = data.iloc[:,-1].values

# Cross validation split
X_train, X_test, y_train, y_test = train_test_split(salary, years_exp, test_size = 0.2, random_state = 0)

# required to reshape into 2D array
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1, 1)

# Fit the linear regression
model = LinearRegression()
model.fit(X_train,y_train)

# prediction
salary_prediction = model.predict(X_test)

plt.scatter(X_train,y_train)
plt.plot(X_test, model.predict(X_test), color = "red")

# export model
joblib.dump(model, "./data/linear_regression_model.pkl")
joblib.dump(X_train, "./data/training_X.pkl")
joblib.dump(y_train, "./data/training_y.pkl")
joblib.dump(X_test, "./data/testing_X.pkl")
joblib.dump(y_test, "./data/testing_y.pkl")