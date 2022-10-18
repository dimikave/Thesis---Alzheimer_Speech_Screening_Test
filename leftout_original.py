from cgi import test
from joblib import PrintTime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, recall_score
import pickle
import warnings
from sklearn.metrics import classification_report
import joblib
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
import random

warnings.filterwarnings('ignore')

def get_train_test_some_split(dfs, train_set_names, test_set_names, people_to_leave):
    
    dfs2 = []

    for df in dfs:
        # print(df.shape)
        dfs2.append(df.copy())
        df.index = df['Name']

    for name in people_to_leave:
        for df in dfs:
            for i, row in df.iterrows():
                if row.Name.startswith(name):
                    df.drop(labels=row.Name, axis=0, inplace=True)
                    # print(df.shape)


    i = 0
    for df in dfs:
        # print(df.iloc[30])
        df = df.reset_index(drop=True)
        df = df.iloc[: , 1:]
        df.to_csv(train_set_names[i])
        i = i+1

    for i in range(5):
        df_train = pd.read_csv(train_set_names[i])
        df_full = dfs2[i]
        # print(~df_full.Name.isin(df_train.Name))
        df_test = df_full[~df_full.Name.isin(df_train.Name)]
        df_test = df_test.iloc[: , 1:]
        df_test.to_csv(test_set_names[i])


df1 = pd.read_csv('Complete_Features_1.csv').dropna().reset_index(drop = True)
df2 = pd.read_csv('Complete_Features_2.csv').dropna().reset_index(drop = True)
df3 = pd.read_csv('Complete_Features_3.csv').dropna().reset_index(drop = True)
df4 = pd.read_csv('Complete_Features_4.csv').dropna().reset_index(drop = True)
df5 = pd.read_csv('Complete_Features_5.csv').dropna().reset_index(drop = True)
dfs = [df1, df2 , df3, df4, df5]

train_names = ['df_some_train_1.csv','df_some_train_2.csv','df_some_train_3.csv','df_some_train_4.csv','df_some_train_5.csv']
test_names = ['df_some_test_1.csv','df_some_test_2.csv','df_some_test_3.csv','df_some_test_4.csv','df_some_test_5.csv']

my_file = open('people_to_leave.txt','r')
people_to_leave = my_file.readlines()
people_to_leave = [i[:-1] for i in people_to_leave]
print(people_to_leave)
get_train_test_some_split(dfs,train_names,test_names,people_to_leave)
