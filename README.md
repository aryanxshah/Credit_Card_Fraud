Credit Card Fraud Detection Project

Overview

This project aims to build a machine learning model to detect fraudulent credit card transactions using logistic regression. The dataset used for this project is a real-world dataset containing transactions made by European cardholders in September 2013. I got this dataset from Kaggle datasets and it is highly imbalanced, with only a small fraction of the transactions being fraudulent.


Dataset

Source: The dataset is available on Kaggle and contains 284,807 transactions.
Columns:
Time: The seconds elapsed between this transaction and the first transaction in the dataset.
V1 to V28: The result of a PCA transformation. Due to confidentiality issues, the original features are not provided.
Amount: The transaction amount.
Class: The target variable, where 1 represents a fraudulent transaction and 0 represents a legitimate transaction.
Project Structure

creditcard.csv: The dataset file containing the transaction data.
creditCardFraud.py: The main Python script that loads the dataset, preprocesses the data, and trains the logistic regression model.
README.md: This file provides an overview of the project and instructions for running the code.
settings.json: Used for changing reference for python3 to python

Prerequisites
To run this project, you need to have Python 3.12 and the following libraries installed:
bash
Copy code
pip install pandas numpy scikit-learn
Steps to Run the Project


The script will load the dataset, preprocess the data, and train a logistic regression model.
The training accuracy and testing accuracy will be printed to the terminal.

Data Preprocessing
Balancing the Dataset: The dataset is  imbalanced, so a random sample of legit transactions was taken to balance the dataset with fraudulent transactions.
Splitting the Dataset: The dataset is split into training and testing sets using an 80-20 split. Stratification is used to ensure the training and testing sets have the same proportion of fraud cases as the original dataset.
Model

Logistic Regression: A simple yet effective algorithm for binary classification tasks like fraud detection.
Evaluation:
Training Accuracy: Measures how well the model fits the training data.
Testing Accuracy: Measures how well the model generalizes to unseen data.
Results

Training Accuracy: 94.79% (example value)
Testing Accuracy: The accuracy on the test set will be displayed after running the script.
Issues and Improvements

Convergence Warning: The logistic regression model may not fully converge within the default number of iterations. Increasing max_iter or scaling the data may help resolve this issue.
Alternative Models: Consider experimenting with other classification models like Random Forest, SVM, or Neural Networks to potentially improve performance.
References

Scikit-Learn Documentation: Logistic Regression
