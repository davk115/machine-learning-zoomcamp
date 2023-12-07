# Midterm Project - Employee Retention Prediction

## Problem Description

This project is to train a model to predict if employees will leave a company.

## Dataset

Dataset used - https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction.

This dataset contains information about employees who worked in a company such as satisfaction levels, salary, number of projects, average monthly hours, tenure and more.

## Training

I first completed the EDA and cleaned up the data. I think started training the models. I used Logistic Regression, Decision Tree, Random Forest and XGBoost. I tuned all models and spend time selecting the correct one. In the end, the XGBoost had the best outcomes and was chosen as the final model.

## Export

I built out a `train.py` script to train the model and save it to a `model.bin` file.

I then created a `predict.py` script to load the model and process the function with employee info using a POST request to determine the prediction score.

I created a `test.py` script to be able to test giving the predict app employee data and then processing it and giving back a determination prediction.

## Setup locally

1. Install Pipenv:
````
pip install pipenv
````


2. Install the requirements:
    - flask
    - numpy
    - scikit-learn==1.3.0
    - gunicorn
    - xgboost
````
pipenv install gunicorn flask numpy scikit-learn==1.3.0 xgboost
````


3. Clone the repo to the Pipenv directory


4. Run the `train.py` script to train the model and export the model file:
````
python train.py
````


5. Run the `predict.py` script to run the webservice using gunicorn and is then ready for input:
````
python predict.py
````


6. You can now run the `test.py` script in another terminal window to pass the employee data to the predict app and get back the termination prediction:
````
python test.py
````

7. Edit the `employee` json in the `test.py` file to be able to test out other variations and get different results.


## Setup using Docker

I've attached screenshots of this process in the `Docker Screenshots` directory.

1. Build a docker image:
````
docker build -t NAME_OF_PROJECT .
````

2. Run the Docker image:
````
docker run -it --rm -p 9696:9696 NAME_OF_PROJECT
````

3. You can now run the `test.py` script in another terminal window to pass the employee data to the predict app and get back the termination prediction:
````
python test.py
````

4. Edit the `employee` json in the `test.py` file to be able to test out other variations and get different results.


## Deploying to the cloud

I used https://render.com to deploy my Docker image.

I created a webservice that can be accessed through https://terminationpred.onrender.com.

**Note, it might take a few seconds to start up if it has been idle.**


You can use this service by editing the url variable in the `test.py` script and changing it to ``https://terminationpred.onrender.com/predict``.