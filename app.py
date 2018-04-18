from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return("The Home Page")

@app.route('/users/<string:username>')
def hello_world(username=None):
    return("Hello {}!".format(username))

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            years_of_experience = float(data["YearsExperience"])
            
            lin_reg = joblib.load("./data/linear_regression_model.pkl")
        except ValueError as e:
            return jsonify("Error in prediction - {}".format(e))

        return jsonify(lin_reg.predict(years_of_experience).tolist())

# @TODO: Alternative process for retraining
# add new data
# create new split?
# test that training & test error rates are similar
#   I.E. abort if everything is all fucked up
@app.route("/retrain", methods=['POST'])
def retrain():
    if request.method == 'POST':
        data = request.get_json()

        try:
            # grab traing information stored on disk
            X_train = joblib.load("./data/training_X.pkl")
            y_train = joblib.load("./data/training_y.pkl")
            X_test = joblib.load("./data/testing_X.pkl")

            # append the new data to the existing data set
            years_exp = np.append(X_train, data['YearsExperience'])
            salary = np.append(y_train, data["Salary"])

            # needs to be in the right format
            years_exp = years_exp.reshape(-1,1)

            # new regression
            new_model = LinearRegression()
            new_model.fit(years_exp, salary)


            # remove old training data
            os.remove("./data/training_X.pkl")
            os.remove("./data/training_y.pkl")

            # generate new training data
            joblib.dump(new_model, "./data/linear_regression_model.pkl")
            joblib.dump(years_exp, "./data/training_X.pkl")
            joblib.dump(salary, "./data/training_y.pkl")

            # sample prediction
            # pred = new_model.predict(X_test)
            # return jsonify(pred.tolist())

        except ValueError as e:
            return jsonify("Error when retraining - {}".format(e))

        return jsonify("Retrained model successfully.")

app.run(host='0.0.0.0', port=8000, debug=True)    
