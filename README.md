# Salary Prediction

![office](office.jpg)
###### **Photo by 'Venveo' on Unsplash**

Simple Implementation of Machine Learning API for modeling salaries with a linear regression. 

Linear Regression is a pretty handy utility, but might not make the perfect algorithm for your salary data set. Many professions show a fair amount of coorelation but others have none what so ever. So consider all of the factors before implementing this application to access your population

### How it works:

1. Upload a dataset csv into the `/data` director
2. Run `python model.py` once to instantiate the model and the cross-validation
3. Run `flask run` to startup the API on `localhost:8000`
4. Post at `/predict` to make a prediction or `/retrain` to add new data to the training data

> Note: The datasets have been redacted so that we can build a production application, but its not too hard to input your own data into the application. 