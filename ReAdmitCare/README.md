# ReAdmitCare

A web application for exploring hospital readmission data and model results.
## Purpose
This project explores hospital readmission using data analysis and machine learning.

## Setup
Install the Python packages with:

    python -m pip install flask pandas joblib scikit-learn xgboost numpy

The dataset folder is kept local and is not included in this public repository.

## Run the app
From the ReAdmitCare project folder, activate the virtual environment and start the app:

    .\.venv\Scripts\Activate.ps1
    python app.py

## Exploratory data analysis (EDA)
The EDA module explores the dataset and creates charts to help examine readmission patterns and patient attributes.

## Data preprocessing
The preprocessing module prepares the data for analysis and model training.

## Classification
The classification module estimates readmission risk using classification models.

## Regression
The regression module estimates hospital stay duration using regression models.

## Model comparison
The project compares model results to help evaluate their performance.

## Feature importance
The feature importance module shows which input variables contribute most to the model results.

## Tree models
The project includes tree-based models for readmission and hospital stay prediction.

## Boosting models
The project includes boosting models for predicting readmission risk and hospital stay.

## Predictions
The prediction page estimates readmission risk and hospital stay duration from patient inputs.

## Model evaluation
The evaluation module reviews prediction performance using metrics and comparison results.

## Model training
The training module fits the project's prediction models using the prepared dataset.

## Model improvement
The improvement module experiments with model settings to improve prediction performance.

## Additional classification models
The project includes additional classifiers for comparing readmission prediction approaches.
