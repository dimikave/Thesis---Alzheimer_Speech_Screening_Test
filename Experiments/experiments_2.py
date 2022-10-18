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

from sklearn.linear_model import LogisticRegression

# ignore all warnings
warnings.filterwarnings('ignore')


###### Parameters
scaling = 0
show_cm = 1
show_feat_imp = 0
exclude_age = 1
stage1_in = 0
ET = 1
# model = ExtraTreesClassifier(class_weight='balanced')
model = ExtraTreesClassifier()
# model = RandomForestClassifier()
# model = svm.SVC(C=15,kernel='rbf')

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

# print(df)

dfs = [df]
i = 0

## Loop through features
for df in dfs:
    Y = df.Diagnosis
    # print(Y.value_counts())

    if exclude_age:
        X = df[df.columns[~df.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age','Stress_Depression','Gender','Education'])]]
    else:
        X = df[df.columns[~df.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr'])]]
        label_1 = LabelEncoder()

        X['Stress_Depression']= label_1.fit_transform(X['Stress_Depression'])
        X['Stress_Depression'] = pd.get_dummies(X['Stress_Depression'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        X['Gender']= label_1.fit_transform(X['Gender'])
        X['Gender'] = pd.get_dummies(X['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        X['Education'] = pd.get_dummies(X['Education'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)

    
    
    X1 = X

    #### SPLIT
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    #### SCALING IF NEED TO
    if scaling:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        X = scaler.transform(X)


    #### FIT MODEL
    model.fit(x_train,y_train)    

    #### PREDICT
    y_pred = model.predict(x_test)

    
    #### ACCURACY , BALANCED ACCURACY, CLASSIFICATION REPORT
    report = classification_report(y_test, y_pred, output_dict=True)
    df_res = pd.DataFrame(report).transpose()
    print('\n\n\n')
    print('--------- Features from all stages ---------- :')
    print(df_res)
    ac = accuracy_score(y_test,y_pred)
    print('Accuracy is: ',ac)
    bac = balanced_accuracy_score(y_test,y_pred)
    print('Balanced Accuracy is: ',bac)

    if ET:
        feat_importances = pd.Series(model.feature_importances_, index=X1.columns)
        feat_importances.nlargest(30).plot(kind='barh')
        most_important_feat = feat_importances.nlargest(30).index.tolist()
        if show_feat_imp:
            plt.show()

    if show_cm:
        ax = plt.subplot()
        cm = confusion_matrix(y_test,y_pred)
        sns.heatmap(cm,annot=True,fmt="d",ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['E-MCI','Healthy','L-MCI','SCD'])
        ax.yaxis.set_ticklabels(['E-MCI','Healthy','L-MCI','SCD'])
        print(y_test.value_counts())
        plt.show()


    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
    n_scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print('Cross-Validated Accuracy : %.3f ± (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

    
    #### REDUCED FEATURES
    if ET:
        x_train_red = x_train[most_important_feat]
        x_test_red = x_test[most_important_feat]
        model.fit(x_train_red,y_train)
        #### PREDICT
        y_pred = model.predict(x_test_red)
    else:
        selector = SelectKBest(f_classif,k=30)
        selector.fit(x_train,y_train)
        important_feat = selector.get_support()
        x_train_s = selector.transform(x_train)
        x_test_s = selector.transform(x_test)
        #### PREDICT
        model.fit(x_train_s,y_train)
        y_pred = model.predict(x_test_s)
    

    if show_cm:
        ax = plt.subplot()
        cm = confusion_matrix(y_test,y_pred)
        sns.heatmap(cm,annot=True,fmt="d",ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['E-MCI','Healthy','L-MCI','SCD'])
        ax.yaxis.set_ticklabels(['E-MCI','Healthy','L-MCI','SCD'])
        print(y_test.value_counts())
        plt.show()


    report = classification_report(y_test, y_pred, output_dict=True)
    df_red_res = pd.DataFrame(report).transpose()
    print('--------------------- Reduced ----------------------')
    print(df_red_res)
    ac = accuracy_score(y_test,y_pred)
    print('Accuracy (Reduced) is: ',ac)
    bac = balanced_accuracy_score(y_test,y_pred)
    print('Balanced Accuracy (Reduced) is: ',bac)

    if ET:
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
        n_scores = cross_val_score(model, X[most_important_feat], Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        print('Cross-Validated Accuracy (Reduced): %.3f ± (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    else:
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
        n_scores = cross_val_score(model, selector.transform(X), Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        print('Cross-Validated Accuracy (Reduced): %.3f ± (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

    print('-----------------------------------------------------')
    i = i+1