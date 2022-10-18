from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import f_classif, chi2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score, classification_report

import warnings

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from statistics import mean
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_score

# ignore all warnings
warnings.filterwarnings('ignore')




def my_loocv(df, model, exclude_aegs, scaling):
    

    ######## INITIALIZE
    acc_scores = []
    rec_scores = []
    pre_scores = []
    f1_scores = []

    acc_scores_r = []
    rec_scores_r = []
    pre_scores_r = []
    f1_scores_r = []

    ######## SPLIT
    df_info = pd.read_csv('alz_info.csv')
    df_full = df.copy()
    ### Loop through names - LOSO / LOOCV
    for name in df_info.Name.to_list():

        df = df_full.copy()
        df.index = df['Name']
        if df_info[df_info['Name']==name].Diagnosis.values[0]=='E-MCI' or df_info[df_info['Name']==name].Diagnosis.values[0]=='L-MCI':
            continue

        for i,row in df.iterrows():
            if row.Name.startswith(name):
                df.drop(labels=row.Name, axis=0, inplace=True)


        df = df.reset_index(drop=True)
        df = df.iloc[: , 1:]
        df_train = df
        df_test = df_full.copy()[~df_full.copy().Name.isin(df_train.Name)]
        df_test = df_test.iloc[: , 1:]
        
        # print(df_test)

        ########## PREPARATION
        df['Diagnosis'] = df['Diagnosis'].replace(['E-MCI'],'MCI')
        df['Diagnosis'] = df['Diagnosis'].replace(['L-MCI'],'MCI')
        df = df[df.Diagnosis != 'MCI']
        Y = df.Diagnosis
        # print(Y.value_counts())
        dft = df_test
        dft['Diagnosis'] = dft['Diagnosis'].replace(['E-MCI'],'MCI')
        dft['Diagnosis'] = dft['Diagnosis'].replace(['L-MCI'],'MCI')
        dft = dft[dft.Diagnosis != 'MCI']
        Y_test = dft.Diagnosis
        # print(dft)
        if exclude_aegs:
            X = df[df.columns[~df.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age','Stress_Depression','Gender','Education'])]]
            X_test = dft[dft.columns[~dft.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age','Stress_Depression','Gender','Education'])]]
        else:
            X = df[df.columns[~df.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr'])]]
            X_test = dft[dft.columns[~dft.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr'])]]
            label_1 = LabelEncoder()

            X['Stress_Depression']= label_1.fit_transform(X['Stress_Depression'])
            X['Stress_Depression'] = pd.get_dummies(X['Stress_Depression'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
            X['Gender']= label_1.fit_transform(X['Gender'])
            X['Gender'] = pd.get_dummies(X['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
            X['Education'] = pd.get_dummies(X['Education'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)

            X_test['Stress_Depression']= label_1.fit_transform(X_test['Stress_Depression'])
            X_test['Stress_Depression'] = pd.get_dummies(X_test['Stress_Depression'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
            X_test['Gender']= label_1.fit_transform(X_test['Gender'])
            X_test['Gender'] = pd.get_dummies(X_test['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
            X_test['Education'] = pd.get_dummies(X_test['Education'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)

        X1 = X
        X1_test = X_test

        #### SCALING IF NEED TO
        if scaling:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_test = scaler.transform(X_test)

        #### FIT MODEL
        model.fit(X,Y)    

        #### PREDICT
        y_pred = model.predict(X_test)
        print(accuracy_score(Y_test,y_pred))
        acc_scores.append(accuracy_score(Y_test,y_pred))
        rec_scores.append(recall_score(Y_test,y_pred, average='weighted'))
        pre_scores.append(precision_score(Y_test,y_pred, average='weighted'))
        f1_scores.append(f1_score(Y_test,y_pred, average='weighted'))
    
        ####### Feature importance
        if ET:
            feat_importances = pd.Series(model.feature_importances_, index=X1.columns)
            # feat_importances.nlargest(30).plot(kind='barh')
            most_important_feat = feat_importances.nlargest(30).index.tolist()
            # print(most_important_feat)
        if ET:
            X_red = X[most_important_feat]
            X_test_red = X_test[most_important_feat]
            model.fit(X_red,Y)
            #### PREDICT
            y_pred = model.predict(X_test_red)
        else:
            selector = SelectKBest(f_classif,k=30)
            selector.fit(X,Y)
            important_feat = selector.get_support()
            X_train_s = selector.transform(X)
            X_test_s = selector.transform(X_test)
            #### PREDICT
            model.fit(X_train_s,Y)
            y_pred = model.predict(X_test_s)
        
        acc_scores_r.append(accuracy_score(Y_test,y_pred))
        rec_scores_r.append(recall_score(Y_test,y_pred, average='weighted'))
        pre_scores_r.append(precision_score(Y_test,y_pred, average='weighted'))
        f1_scores_r.append(f1_score(Y_test,y_pred, average='weighted'))
    


    print('-------- LOOCV RESULTS --------- :')
    print('Accuracy: ', np.mean(acc_scores))
    print('Precision: ', np.mean(pre_scores))
    print('Recall: ', np.mean(rec_scores))
    print('F1-score: ', np.mean(f1_scores))
    print('\n\n\n')
    print('-------- LOOCV RESULTS after feature selection --------- :')
    print('Accuracy: ', np.mean(acc_scores_r))
    print('Precision: ', np.mean(pre_scores_r))
    print('Recall: ', np.mean(rec_scores_r))
    print('F1-score: ', np.mean(f1_scores_r))
        
    





###### Parameters
stage1_in = 1
ET = 1
exclude_aegs = 1
scaling = 0


# model = ExtraTreesClassifier(class_weight='balanced')
model = ExtraTreesClassifier()
# model = RandomForestClassifier(class_weight='balanced')
# model = RandomForestClassifier(n_estimators=1000)
# model = svm.SVC(C=1,kernel='rbf')

## Load files from original features
df1 = pd.read_csv('Complete_Features_1.csv').dropna().reset_index(drop = True)
df2 = pd.read_csv('Complete_Features_2.csv').dropna().reset_index(drop = True)
df3 = pd.read_csv('Complete_Features_3.csv').dropna().reset_index(drop = True)
df4 = pd.read_csv('Complete_Features_4.csv').dropna().reset_index(drop = True)
df5 = pd.read_csv('Complete_Features_5.csv').dropna().reset_index(drop = True)

if stage1_in:
    df = df1.append(df2,ignore_index=True)
    df = df.append(df3,ignore_index=True)
    df = df.append(df4,ignore_index=True)
    df = df.append(df5,ignore_index=True)

else:
    df = df2.append(df3,ignore_index=True)
    df = df.append(df4,ignore_index=True)
    df = df.append(df5,ignore_index=True)




my_loocv(df,model,exclude_aegs,scaling)

