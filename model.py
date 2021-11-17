
#1. Importing dependent libraries

import pandas as pd
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


#2. Loading the Dataset

df = pd.read_csv('../transactions_train/transactions_train.csv')

#3.Data Preprocessing

X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]

Y = X['isFraud']
del X['isFraud']

# Eliminating columns irrelevant for model
X = X.drop(['nameOrig', 'nameDest'], axis = 1)

# Binary-encoding of labelled data in 'type'
X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int) 

X.loc[(X.oldbalanceDest == 0) & (X.newbalanceDest == 0) & (X.amount != 0), ['oldbalanceDest', 'newbalanceDest']] = -1
X.loc[(X.oldbalanceOrig == 0) & (X.newbalanceOrig == 0) & (X.amount != 0), ['oldbalanceOrig', 'newbalanceOrig']] = np.nan

X['errorbalanceOrig'] = X.newbalanceOrig + X.amount - X.oldbalanceOrig
X['errorbalanceDest'] = X.oldbalanceDest + X.amount - X.newbalanceDest

#XG boost classifier

randomState = 5
np.random.seed(randomState)

trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2, random_state = randomState)
weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())

clf = XGBClassifier(max_depth = 3, scale_pos_weight = weights, n_jobs = 4)
model = clf.fit(trainX, trainY)

# probabilities = clf.fit(trainX, trainY).predict_proba(testX)
# ypred = clf.fit(trainX, trainY).predict(testX)
# print('AUPRC = {}'.format(average_precision_score(testY, probabilities[:, 1])))


#Saving the model

# from sklearn.externals import joblib
import joblib

joblib.dump(model, 'model.pkl')
print("Model dumped!")

# Loading the model saved
model = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(trainX.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")