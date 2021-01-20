# -*- coding: utf-8 -*-
"""Fraud detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1s3y7YlQajhOPd7XxodS-HWVnpq-bNlMc
"""

! pip install -q kaggle
from google.colab import files
files.upload() #upload kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls ~/.kaggle
!chmod 600 /root/.kaggle/kaggle.json
!mkdir -p /content/unzip_file

# !kaggle kernels list — user YOUR_USER — sort-by dateRun
!kaggle competitions download -c ieee-fraud-detection
!unzip -q train_identity.csv.zip -d /content/unzip_file/
!unzip -q train_transaction.csv.zip -d /content/unzip_file/
!unzip -q test_identity.csv.zip -d /content/unzip_file/
!unzip -q test_transaction.csv.zip -d /content/unzip_file/
!ls

#!pip install matplotlib-venn
#!apt-get -qq install -y libfluidsynth1
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
sns.set()

train_identity = pd.read_csv('/content/unzip_file/train_identity.csv')
train_transaction = pd.read_csv('/content/unzip_file/train_transaction.csv')

test_identity = pd.read_csv('/content/unzip_file/test_identity.csv')
test_transaction = pd.read_csv('/content/unzip_file/test_transaction.csv')

def downcast_dtype(df):
    _start = df.memory_usage(deep=True).sum() / 1024 ** 2
    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype in ['int64', 'int32']]
    df[float_cols] = df[float_cols].astype('float32')
    df[int_cols] = df[int_cols].astype('int32')
    _end = df.memory_usage(deep=True).sum() / 1024 ** 2
    saved = (_start - _end) / _start * 100
    print(f"Save {saved:.2f}%")
    return df

train_identity = downcast_dtype(train_identity)
test_identity = downcast_dtype(test_identity)
train_transaction = downcast_dtype(train_transaction)
test_transaction = downcast_dtype(test_transaction)

train_transaction.head()

test_transaction.head()

train_identity.columns

print("train_identity.size", train_identity.size)
print("test_identity.size", test_identity.size)
print("train_identity.shape", train_identity.shape)
print("test_identity.shape", test_identity.shape)
print("train_transaction.size", train_transaction.size)
print("test_transaction.size", test_transaction.size)
print("train_transaction.shape", train_transaction.shape)
print("test_transaction.shape", test_transaction.shape)

data_inmerge = train_transaction[['TransactionID', 'isFraud']].copy()

data_inmerge.shape

"""#Merging two dataframe"""

train_identity = pd.merge(left=train_identity, right=data_inmerge, how='inner', on='TransactionID', left_index=True)
print('train_identity_shape = ', train_identity.shape)
#print('test_data_shape = ', test_data.shape)

train_identity.head()

print(train_transaction.info())
print(train_identity.info())

train_transaction.isna().sum()

train_identity.isna().sum()

train_transaction = train_transaction.dropna(axis = 1, thresh =(len(train_transaction)-100000))
train_identity = train_identity.dropna(axis=1, thresh=(len(train_identity)-70000))
print("train_transaction.shape", train_transaction.shape)
print("train_identity.shape", train_identity.shape)

def binary_columns(df):
  columns_num = list(df.select_dtypes(include=['float32','int32']).columns)
  for i in df[columns_num]:
    if max(df[i]) == 1:
      yield i
    else:
      pass

binary_columns_tr = list(binary_columns(train_transaction))
binary_columns_id = list(binary_columns(train_identity))

print("binary columns in train transaction data is", binary_columns_tr)
print('binay columns in train identity data is', binary_columns_id)

transaction_corr = train_transaction.corr(method ='pearson')
identity_corr = train_identity.corr(method='pearson')

print("transaction correlation shape: ", transaction_corr.shape)
transaction_corr

print("identity correlaton shape: ", identity_corr.shape)
identity_corr

TcorrwithTarget = train_transaction.corrwith(train_transaction['isFraud'])
plt.figure(figsize=(16, 6))
plt.plot(TcorrwithTarget.index, TcorrwithTarget.values, 'r--')

IcorrwithTarget = train_identity.corrwith(train_identity['isFraud'])
plt.figure(figsize=(16, 6))
plt.plot(IcorrwithTarget.index, IcorrwithTarget.values, 'r--')

"""#Remove highly correlated features"""

def remove_redundancies(corr):
  correlated_features = set()
  correlation_matrix = corr
  for i in range(len(correlation_matrix .columns)):
      for j in range(i):
          if abs(corr.iloc[i, j]) > 0.9:
              colname = corr.columns[i]
              correlated_features.add(colname)
  return correlated_features

Tcorrelated_features = remove_redundancies(transaction_corr)
Icorrelated_features = remove_redundancies(identity_corr)

train_transaction.drop(labels=Tcorrelated_features, axis=1, inplace=True)
train_identity.drop(labels = Icorrelated_features, axis =1, inplace=True)
#test_features.drop(labels=correlated_features, axis=1, inplace=True)

print("train_transaction.shape ", train_transaction.shape)
print("train_identity.shape ", train_identity.shape)

TcorrwithTarget = train_transaction.corrwith(train_transaction['isFraud'])
plt.figure(figsize=(16, 6))
plt.plot(TcorrwithTarget.index, TcorrwithTarget.values, 'r--')

IcorrwithTarget = train_identity.corrwith(train_identity['isFraud'])
plt.figure(figsize=(16, 6))
plt.plot(IcorrwithTarget.index, IcorrwithTarget.values, 'r--')

"""#Fill null values"""

def colwithnul(df):
  col_nem = list(df.select_dtypes(include=['float32','int32']).columns)
  #col_nem.remove('TransactionID')
  #col_nem.remove('isFraud')
  return col_nem

Tcolnum = colwithnul(train_transaction)
Icolnum = colwithnul(train_identity)

# Columns which gonna fill with mean
def fillnullmean(df, colnum, binary_column):
  colnum = [item for item in colnum if item not in binary_column]
  for i in colnum:
    z_high = []
    z_score = stats.zscore(df[i])
    [z_high.append(a) for a in abs(z_score) if a > 3]
    if len(z_high) < 5:
      #print(len(z_high))
      df[i].fillna(df[i].mean(), inplace=True)
    else:
      print(i, len(z_high))
      pass

# Columns which gonna fill with mode
def fillnullmod(df, binary_column):
  binary_column.remove('isFraud')
  for i in binary_column:
    z_high = []
    z_score = stats.zscore(df[i])
    [z_high.append(a) for a in abs(z_score) if a > 3]
    if len(z_high) < 5:
      #print(len(z_high))
      df[i].fillna(df[i].mode(), inplace=True)
    else:
      print(i, len(z_high))
      pass

fillnullmean(df=train_transaction, colnum=Tcolnum, binary_column=binary_columns_tr)
fillnullmean(df=train_identity, colnum=Icolnum, binary_column=binary_columns_id)

fillnullmod(df=train_transaction, binary_column=binary_columns_tr)
fillnullmod(df=train_identity, binary_column=binary_columns_id)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(train_transaction.isna().sum())

train_identity.isna().sum()

plt.figure(figsize=(16, 6))
sns.barplot(x='card4', y='isFraud', data=train_transaction)
#plt.bar(train_transaction['card4'].unique(), train_transaction['isFraud'])

plt.figure(figsize=(16, 6))
sns.scatterplot(x='TransactionAmt', y='isFraud', data=train_transaction)

print(train_transaction['card4'].dtypes)
print(train_transaction['P_emaildomain'].dtypes)

plt.figure(figsize=(16,6))
sns.barplot(x='P_emaildomain', y='isFraud', data=train_transaction)

fig_dims = (8, 6)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(train_transaction.isnull(), cbar=False)

train_transaction.head(10)

plt.plot(train_transaction['TransactionAmt'], train_transaction['isFraud'], "bo")

sns.boxplot(x = train_transaction['TransactionAmt'])

sns.boxplot(x = train_transaction['card1'])

[(index, value) for index, value  in corrwithTarget.items() if value > .2]