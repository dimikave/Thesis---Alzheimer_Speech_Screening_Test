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

    for i in range(10):
        df_train = pd.read_csv(train_set_names[i])
        df_full = dfs2[i]
        # print(~df_full.Name.isin(df_train.Name))
        df_test = df_full[~df_full.Name.isin(df_train.Name)]
        df_test = df_test.iloc[: , 1:]
        df_test.to_csv(test_set_names[i])



df_21 = pd.read_csv('New_feat_complete_21.csv')
df_41 = pd.read_csv('New_feat_complete_41.csv')
df_31 = pd.read_csv('New_feat_complete_31.csv')
df_51 = pd.read_csv('New_feat_complete_51.csv')

df_32 = pd.read_csv('New_feat_complete_32.csv')
df_42 = pd.read_csv('New_feat_complete_42.csv')
df_52 = pd.read_csv('New_feat_complete_52.csv')

df_43 = pd.read_csv('New_feat_complete_43.csv')
df_53 = pd.read_csv('New_feat_complete_53.csv')

df_54 = pd.read_csv('New_feat_complete_54.csv')


dfs = [df_21, df_31, df_41, df_51, df_32, df_42, df_52, df_43, df_53, df_54]
train_names = ['df_some_train_21.csv','df_some_train_31.csv','df_some_train_41.csv','df_some_train_51.csv','df_some_train_32.csv','df_some_train_42.csv','df_some_train_52.csv','df_some_train_43.csv','df_some_train_53.csv','df_some_train_54.csv']
test_names = ['df_some_test_21.csv','df_some_test_31.csv','df_some_test_41.csv','df_some_test_51.csv','df_some_test_32.csv','df_some_test_42.csv','df_some_test_52.csv','df_some_test_43.csv','df_some_test_53.csv','df_some_test_54.csv']
my_file = open('people_to_leave.txt','r')
people_to_leave = my_file.readlines()
people_to_leave = [i[:-1] for i in people_to_leave]
print(people_to_leave)
get_train_test_some_split(dfs,train_names,test_names,people_to_leave)
