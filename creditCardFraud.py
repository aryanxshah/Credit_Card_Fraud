import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the dataset to Pandas DataFrame
credit_card_data = pd.read_csv('/Users/aryan/GitHub Projects/Credit Card Project/creditcard.csv')

#test readability first five rows of csv file
credit_card_data.head()

#read for information
credit_card_data.info()

#check for missing values in each column
credit_card_data.isnull().sum()

#distribution of legit (0)/fraud (1) transactions 
credit_card_data['Class'].value_counts()
legitData = credit_card_data[credit_card_data.Class == 0]
fraudData = credit_card_data[credit_card_data.Class == 1]
print(legitData.shape)
print(fraudData.shape)

#statistical measures of the data and compare mean value of columns
legitData.Amount.describe()
fraudData.Amount.describe()
credit_card_data.groupby('Class').mean()

#create sample data with similar distributions of legit(1) and fraud(0) transactions
legitSample = legitData.sample(n=492)
newDataset = pd.concat([legitSample, fraudData], axis =0)

newDataset.head()
newDataset.tail()

newDataset['Class'].value_counts()
newDataset.groupby('Class').mean()

X = newDataset.drop(columns = 'Class', axis = 1)
Y = newDataset['Class']

print(X)
print(Y)

#split data into trianing data and testing data (ML modeling)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#check splits before training model
print(X.shape, X_train.shape, X_test.shape)

#Train and evaluate the ogistic regression model training using the training Data 
model = LogisticRegression()
model.fit(X_train, Y_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training Data: ', training_data_accuracy)

#test data accuracy
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Training Data: ', test_data_accuracy)
