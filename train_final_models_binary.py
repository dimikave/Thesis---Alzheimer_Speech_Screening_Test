import numpy as np
import matplotlib.pyplot as plt
from pydantic import Extra
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

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
import joblib

warnings.filterwarnings('ignore')



### Train Models
def train_model(model, df, df_test, save_model_name, save_scaler_name, age_flag, education_flag, gender_flag, scaling, binary):

    ### Drop NaN and Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna().reset_index(drop = True)

    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test = df_test.fillna(0).reset_index(drop = True)
    dft = df_test

    # Drop according to binary
    if binary == 'hs':
        df['Diagnosis'] = df['Diagnosis'].replace(['E-MCI'],'MCI')
        df['Diagnosis'] = df['Diagnosis'].replace(['L-MCI'],'MCI')
        df = df[df.Diagnosis != 'MCI']  
        Y = df.Diagnosis
        dft['Diagnosis'] = dft['Diagnosis'].replace(['E-MCI'],'MCI')
        dft['Diagnosis'] = dft['Diagnosis'].replace(['L-MCI'],'MCI')
        dft = dft[dft.Diagnosis != 'MCI'] 
        Y_test = dft.Diagnosis
    
    elif binary == 'hm':
        df['Diagnosis'] = df['Diagnosis'].replace(['E-MCI'],'MCI')
        df['Diagnosis'] = df['Diagnosis'].replace(['L-MCI'],'MCI')
        df = df[df.Diagnosis != 'SCD']   
        dft['Diagnosis'] = dft['Diagnosis'].replace(['E-MCI'],'MCI')
        dft['Diagnosis'] = dft['Diagnosis'].replace(['L-MCI'],'MCI')
        dft = dft[dft.Diagnosis != 'SCD'] 
        Y = df.Diagnosis
        Y_test = dft.Diagnosis
    elif binary == 'sm':
        df['Diagnosis'] = df['Diagnosis'].replace(['E-MCI'],'MCI')
        df['Diagnosis'] = df['Diagnosis'].replace(['L-MCI'],'MCI')
        df = df[df.Diagnosis != 'Healthy']   
        dft['Diagnosis'] = dft['Diagnosis'].replace(['E-MCI'],'MCI')
        dft['Diagnosis'] = dft['Diagnosis'].replace(['L-MCI'],'MCI')
        dft = dft[dft.Diagnosis != 'Healthy'] 
        Y = df.Diagnosis
        Y_test = dft.Diagnosis


    ### Prepare the data-set
    print(Y_test.value_counts())
    label_1 = LabelEncoder()

    if age_flag and education_flag and gender_flag:
        X = df[df.columns[~df.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr'])]]
        X['Gender']= label_1.fit_transform(X['Gender'])
        X['Gender'] = pd.get_dummies(X['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        X['Education'] = pd.get_dummies(X['Education'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        X_test = dft[dft.columns[~dft.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr'])]]
        X_test['Gender']= label_1.transform(X_test['Gender'])
        X_test['Gender'] = pd.get_dummies(X_test['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        X_test['Education'] = pd.get_dummies(X_test['Education'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        
        s = '_AEG.sav'

    elif education_flag and gender_flag:
        X = df[df.columns[~df.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age'])]]
        X['Gender']= label_1.fit_transform(X['Gender'])
        X['Gender'] = pd.get_dummies(X['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        X['Education'] = pd.get_dummies(X['Education'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        X_test = dft[dft.columns[~dft.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age'])]]
        X_test['Gender']= label_1.transform(X_test['Gender'])
        X_test['Gender'] = pd.get_dummies(X_test['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        X_test['Education'] = pd.get_dummies(X_test['Education'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        s = '_EG.sav'
    elif gender_flag:
        X = df[df.columns[~df.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age','Education'])]]
        X['Gender']= label_1.fit_transform(X['Gender'])
        X['Gender'] = pd.get_dummies(X['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        X_test = dft[dft.columns[~dft.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age','Education'])]]
        X_test['Gender']= label_1.transform(X_test['Gender'])
        X_test['Gender'] = pd.get_dummies(X_test['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        s = '_G.sav'
    else:
        X = df[df.columns[~df.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age','Education','Gender'])]]
        X_test = dft[dft.columns[~dft.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age','Education','Gender'])]]
        s = '.sav'

    ### Encoding
    X['Stress_Depression']= label_1.fit_transform(X['Stress_Depression'])
    X['Stress_Depression'] = pd.get_dummies(X['Stress_Depression'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
    X_test['Stress_Depression']= label_1.transform(X_test['Stress_Depression'])
    X_test['Stress_Depression'] = pd.get_dummies(X_test['Stress_Depression'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
    X_test_1 = X_test
    ### Train-test split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    ### Scaling if need to
    if scaling:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        X = scaler.transform(X)
        X_test = scaler.transform(X_test)
        joblib.dump(scaler, save_scaler_name)
    

    ### Model
    clf = model
    clf.fit(x_train, y_train)
    

    # feat_importances = pd.Series(clf.feature_importances_, index=X_test_1.columns)
    # feat_importances.nlargest(30).plot(kind='barh')
    # most_important_feat = feat_importances.nlargest(30).index.tolist()
    # plt.show()

    ### Print model metrics
    if binary == 'hs':
        print('Classification Report for Train Set: ')
        print(classification_report(y_test, clf.predict(x_test), target_names=['Healthy','SCD']))
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
        n_scores = cross_val_score(clf, X, Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        print('Cross-Validated Accuracy : %.3f ± (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

        print('Classification Report for Test Set: ')
        print(Y_test.shape)
        print(X_test.shape)
        print(classification_report(Y_test, clf.predict(X_test), target_names=['Healthy','SCD']))
    
    elif binary == 'hm':
        print('Classification Report for Train Set: ')
        print(classification_report(y_test, clf.predict(x_test), target_names=['Healthy','MCI']))
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
        n_scores = cross_val_score(clf, X, Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        print('Cross-Validated Accuracy : %.3f ± (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

        print('Classification Report for Test Set: ')
        print(classification_report(Y_test, clf.predict(X_test), target_names=['Healthy','MCI']))
    
    elif binary == 'sm':
        print('Classification Report for Train Set: ')
        print(classification_report(y_test, clf.predict(x_test), target_names=['MCI','SCD']))
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
        n_scores = cross_val_score(clf, X, Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        print('Cross-Validated Accuracy : %.3f ± (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

        print('Classification Report for Test Set: ')
        print(classification_report(Y_test, clf.predict(X_test), target_names=['MCI','SCD']))
    
        

    # Save the model to disk
    filename = save_model_name+s
    pickle.dump(model, open(filename, 'wb'))




## Load data sets

df1_train = pd.read_csv('df_some_train_1.csv').dropna().reset_index(drop = True)
df2_train = pd.read_csv('df_some_train_2.csv').dropna().reset_index(drop = True)
df3_train = pd.read_csv('df_some_train_3.csv').dropna().reset_index(drop = True)
df4_train = pd.read_csv('df_some_train_4.csv').dropna().reset_index(drop = True)
df5_train = pd.read_csv('df_some_train_5.csv').dropna().reset_index(drop = True)

dfs_train = [df1_train, df2_train, df3_train, df4_train, df5_train]

df1_test = pd.read_csv('df_some_test_1.csv').dropna().reset_index(drop = True)
df2_test = pd.read_csv('df_some_test_2.csv').dropna().reset_index(drop = True)
df3_test = pd.read_csv('df_some_test_3.csv').dropna().reset_index(drop = True)
df4_test = pd.read_csv('df_some_test_4.csv').dropna().reset_index(drop = True)
df5_test = pd.read_csv('df_some_test_5.csv').dropna().reset_index(drop = True)

dfs_test = [df1_test, df2_test, df3_test, df4_test, df5_test]


# models = [svm.SVC(kernel='rbf',C=10,probability=True),svm.SVC(kernel='rbf',C=20,probability=True), svm.SVC(kernel='rbf',C=12,probability=True), svm.SVC(kernel='rbf',C=14,probability=True), svm.SVC(kernel='rbf',C=14,probability=True), svm.SVC(kernel='rbf',C=14,probability=True), svm.SVC(kernel='rbf',C=14,probability=True), svm.SVC(kernel='rbf',C=10,probability=True), svm.SVC(kernel='rbf',C=25,probability=True), svm.SVC(kernel='rbf',C=10,probability=True),]
models = [ExtraTreesClassifier(),ExtraTreesClassifier(),ExtraTreesClassifier(), ExtraTreesClassifier(),ExtraTreesClassifier()]


################ Some people out Testing 5 models:
# model_names = ['SVM_1_hs','SVM_2_hs','SVM_3_hs','SVM_4_hs','SVM_5_hs']
# scaler_names = ['Scaler_1_hs.gz','Scaler_2_hs.gz','Scaler_3_hs.gz','Scaler_4_hs.gz','Scaler_5_hs.gz']

model_names = ['SVM_1_hm','SVM_2_hm','SVM_3_hm','SVM_4_hm','SVM_5_hm']
scaler_names = ['Scaler_1_hm.gz','Scaler_2_hs.gz','Scaler_3_hm.gz','Scaler_4_hm.gz','Scaler_5_hm.gz']

# model_names = ['SVM_1_sm','SVM_2_sm','SVM_3_sm','SVM_4_sm','SVM_5_sm']
# scaler_names = ['Scaler_1_sm.gz','Scaler_2_sm.gz','Scaler_3_sm.gz','Scaler_4_sm.gz','Scaler_5_sm.gz']




for i in range(5):
    train_model(models[i], dfs_train[i], dfs_test[i], model_names[i], scaler_names[i], age_flag=False, education_flag=False, gender_flag=False, scaling=True, binary='hm')
    # train_model(models[i], dfs_train[i], dfs_test[i], model_names[i], scaler_names[i], age_flag=True, education_flag=True, gender_flag=True, scaling=True)
    # train_model(models[i], dfs_train[i], dfs_test[i], model_names[i], scaler_names[i], age_flag=True, education_flag=True, gender_flag=True, scaling=False)
    # train_model(models[i], dfs_train[i], dfs_test[i], model_names[i], scaler_names[i], age_flag=False, education_flag=False, gender_flag=False, scaling=False)
