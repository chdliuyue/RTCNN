import pandas as pd
import pickle
from models_pytorch.utils import cats2ints, cats2ints_transform
import numpy as np

# load training data
with open('data/X_TRAIN.pkl', 'rb') as fp:
    X_TRAIN = pickle.load(fp)

Q_df_TRAIN = pd.read_csv('data/Q_train.csv')

with open('data/y_TRAIN.pkl', 'rb') as fp:
    y_TRAIN = pickle.load(fp)

# Load the test data:
with open('data/X_TEST.pkl', 'rb') as fp:
    X_TEST = pickle.load(fp)

Q_df_TEST = pd.read_csv('data/Q_test.csv')

with open('data/y_TEST.pkl', 'rb') as fp:
    y_TEST = pickle.load(fp)


# X size is (num, var, choice, 1) 可能是每个样本三种都做过
# Continuous variables (X) (plus intercepts) ? 加这个的目的是什么
X_vars = ['ASC_Car',
          'ASC_SM',
          'TT_SCALED(/100)',
          'COST_SCALED(/100)',
          'Headway_Train_SM']

# Categorical variables (Q) to be encoded as embeddings
Q_vars = ['PURPOSE', 'FIRST', 'TICKET', 'WHO',
          'LUGGAGE', 'AGE', 'MALE', 'INCOME',
          'GA', 'ORIGIN', 'DEST', 'SM_SEATS']

# print(Q_df_TRAIN.head())
NUM_OBS_TRAIN = X_TRAIN.shape[0]
NUM_OBS_TEST = X_TEST.shape[0]

NUM_CHOICES = X_TRAIN.shape[2]
NUM_X_VARS = X_TRAIN.shape[1]
NUM_Q_VARS = Q_df_TRAIN.shape[-1]

UNIQUE_CATS = sorted(list(set(Q_df_TRAIN.values.reshape(-1))))
NUM_UNIQUE_CATS = len(UNIQUE_CATS)

print('The number of alternatives (C) is:', NUM_CHOICES)
print('The number of continuous variables (|X|) is:', NUM_X_VARS)
print('The number of categorical variables (|Q|) to be encoded as embeddings is:', NUM_Q_VARS)
print('The number of unique categories in Q (Z) is:', NUM_UNIQUE_CATS)
print()
print('The number of observations in the training set is:', NUM_OBS_TRAIN)
print('The number of observations in the test set is:', NUM_OBS_TEST)


cats2ints_mapping = cats2ints(Q_df_TRAIN)
print(cats2ints_mapping)

Q_TRAIN = cats2ints_transform(Q_df_TRAIN, cats2ints_mapping)
Q_TEST = cats2ints_transform(Q_df_TEST, cats2ints_mapping)

# print(Q_TRAIN)
# (n_obs, n_variables in Q)
# print(X_TRAIN.shape)
# print(Q_TRAIN.shape)
# print(y_TRAIN.shape)
