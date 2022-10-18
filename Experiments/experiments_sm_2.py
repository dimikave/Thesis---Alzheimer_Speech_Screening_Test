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
scaling = 1
show_cm = 0
show_feat_imp = 0
exclude_age = 1
stage1_in = 1
ET = 0

# model = ExtraTreesClassifier(class_weight='balanced')
# model = ExtraTreesClassifier()
# model = RandomForestClassifier(class_weight='balanced')
# model = RandomForestClassifier(n_estimators=1000)
model = svm.SVC(C=1,kernel='rbf')

## Load files from original features
df1_train = pd.read_csv('df_some_train_1.csv').dropna().reset_index(drop = True)
df2_train = pd.read_csv('df_some_train_2.csv').dropna().reset_index(drop = True)
df3_train = pd.read_csv('df_some_train_3.csv').dropna().reset_index(drop = True)
df4_train = pd.read_csv('df_some_train_4.csv').dropna().reset_index(drop = True)
df5_train = pd.read_csv('df_some_train_5.csv').dropna().reset_index(drop = True)

df1_test = pd.read_csv('df_some_test_1.csv').dropna().reset_index(drop = True)
df2_test = pd.read_csv('df_some_test_2.csv').dropna().reset_index(drop = True)
df3_test = pd.read_csv('df_some_test_3.csv').dropna().reset_index(drop = True)
df4_test = pd.read_csv('df_some_test_4.csv').dropna().reset_index(drop = True)
df5_test = pd.read_csv('df_some_test_5.csv').dropna().reset_index(drop = True)

if stage1_in:
    df_train = df1_train.append(df2_train,ignore_index=True)
    df_train = df_train.append(df3_train,ignore_index=True)
    df_train = df_train.append(df4_train,ignore_index=True)
    df_train = df_train.append(df5_train,ignore_index=True)

    df_test = df1_test.append(df2_test,ignore_index=True)
    df_test = df_test.append(df3_test,ignore_index=True)
    df_test = df_test.append(df4_test,ignore_index=True)
    df_test = df_test.append(df5_test,ignore_index=True)

else:
    df_train = df2_train.append(df3_train,ignore_index=True)
    df_train = df_train.append(df4_train,ignore_index=True)
    df_train = df_train.append(df5_train,ignore_index=True)

    df_test = df2_test.append(df3_test,ignore_index=True)
    df_test = df_test.append(df4_test,ignore_index=True)
    df_test = df_test.append(df5_test,ignore_index=True)

# print(df)

dfs = [df_train]
dfs_test= [df_test]

i = 0

## Loop through features
for df in dfs:
    df = df.drop(columns=['Unnamed: 0'])
    df['Diagnosis'] = df['Diagnosis'].replace(['E-MCI'],'MCI')
    df['Diagnosis'] = df['Diagnosis'].replace(['L-MCI'],'MCI')
    df = df[df.Diagnosis != 'Healthy']
    df = df.reset_index(drop=True)
    Y = df.Diagnosis
    # print(Y.value_counts())
    dft = dfs_test[i]
    dft = dft.drop(columns=['Unnamed: 0'])
    dft['Diagnosis'] = dft['Diagnosis'].replace(['E-MCI'],'MCI')
    dft['Diagnosis'] = dft['Diagnosis'].replace(['L-MCI'],'MCI')
    dft = dft[dft.Diagnosis != 'Healthy']
    dft = dft.reset_index(drop=True)
    Y_test = dft.Diagnosis
    if exclude_age:
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

    #### SPLIT
    ## Split already done from using leftout_original.py and using the splitted df files

    #### SCALING IF NEED TO
    if scaling:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)


    #### FIT MODEL
    model.fit(X,Y)    

    #### PREDICT
    y_pred = model.predict(X_test)

    #### SHOW MISCLASSIFICATIONS
    hit_dict = {}
    Names = dft.Name.to_list()
    Names_new = []
    
    for name in Names:
        new_name = name.rpartition('_')[0]
        new_name = new_name.rpartition('_')[0]
        Names_new.append(new_name)

    # unique_names = np.unique(np.array(Names_new))
    # print(unique_names)
    Names = pd.Series(Names_new)
    j = 0
    print(len(Y_test))
    for name in Names_new:
        if name not in hit_dict:
            hit_dict[name] = 0
        if y_pred[j]==Y_test[j]:
            # print(j)
            hit_dict[name] = hit_dict[name] + 1
        j = j+1

    counts = pd.Series(Names_new).value_counts()
    counts = counts.to_dict()
    scores = {}
    for key in hit_dict:
        scores[key] = hit_dict[key]/counts[key] * 100
    last_dict = {'Hit Ratio': list(scores.values()), 'Counter': list(counts.values())}
    scores_pd = pd.DataFrame(data=last_dict, index= list(scores.keys()))
    diags = []
    df_info = pd.read_csv('alz_info.csv')
    for name in list(scores.keys()):
        print(name)
        if name in df_info.Name.to_list():
            diags.extend(df_info[df_info['Name'] == name].Diagnosis.to_list())
    # print(diags)
    scores_pd['Diagnosis'] = diags
    scores_pd['Diagnosis'] = scores_pd['Diagnosis'].replace(['E-MCI'],'MCI')
    scores_pd['Diagnosis'] = scores_pd['Diagnosis'].replace(['L-MCI'],'MCI')
    print(scores_pd)
    # print(df)
    # print(Names)

    #### ACCURACY , BALANCED ACCURACY, CLASSIFICATION REPORT
    report = classification_report(Y_test, y_pred, output_dict=True)
    df_res = pd.DataFrame(report).transpose()
    print('\n\n\n')
    print('--------- Features from all stages ---------- :')
    print(df_res)
    ac = accuracy_score(Y_test,y_pred)
    print('Accuracy is: ',ac)
    bac = balanced_accuracy_score(Y_test,y_pred)
    print('Balanced Accuracy is: ',bac)

    if ET:
        feat_importances = pd.Series(model.feature_importances_, index=X1.columns)
        feat_importances.nlargest(30).plot(kind='barh')
        most_important_feat = feat_importances.nlargest(30).index.tolist()
        if show_feat_imp:
            plt.show()

    if show_cm:
        ax = plt.subplot()
        cm = confusion_matrix(Y_test,y_pred)
        sns.heatmap(cm,annot=True,fmt="d",ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['MCI','SCD'])
        ax.yaxis.set_ticklabels(['MCI','SCD'])
        # print(Y_test.value_counts())
        plt.show()


    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
    # n_scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # print('Cross-Validated Accuracy : %.3f ± (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

    
    #### REDUCED FEATURES
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
    

    if show_cm:
        ax = plt.subplot()
        cm = confusion_matrix(Y_test,y_pred)
        sns.heatmap(cm,annot=True,fmt="d",ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['MCI','SCD'])
        ax.yaxis.set_ticklabels(['MCI','SCD'])
        # print(Y_test.value_counts())
        plt.show()


    report = classification_report(Y_test, y_pred, output_dict=True)
    df_red_res = pd.DataFrame(report).transpose()
    print('--------------------- Reduced ----------------------')
    print(df_red_res)
    ac = accuracy_score(Y_test,y_pred)
    print('Accuracy (Reduced) is: ',ac)
    bac = balanced_accuracy_score(Y_test,y_pred)
    print('Balanced Accuracy (Reduced) is: ',bac)

    # if ET:
    #     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
    #     n_scores = cross_val_score(model, X[most_important_feat], Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    #     print('Cross-Validated Accuracy (Reduced): %.3f ± (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    # else:
    #     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
    #     n_scores = cross_val_score(model, selector.transform(X), Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    #     print('Cross-Validated Accuracy (Reduced): %.3f ± (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

    print('-----------------------------------------------------')
    i = i+1