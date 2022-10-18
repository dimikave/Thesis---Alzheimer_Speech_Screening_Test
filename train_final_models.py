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
def train_model(model, df, df_test, save_model_name, save_scaler_name, age_flag, education_flag, gender_flag, scaling):

    ### Drop NaN and Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna().reset_index(drop = True)

    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test = df_test.fillna(0).reset_index(drop = True)

    ### Prepare the data-set
    Y = df.Diagnosis
    Y_test = df_test.Diagnosis
    print(Y_test)
    label_1 = LabelEncoder()

    if age_flag and education_flag and gender_flag:
        X = df[df.columns[~df.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr'])]]
        X['Gender']= label_1.fit_transform(X['Gender'])
        X['Gender'] = pd.get_dummies(X['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        X['Education'] = pd.get_dummies(X['Education'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        X_test = df_test[df_test.columns[~df_test.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr'])]]
        X_test['Gender']= label_1.transform(X_test['Gender'])
        X_test['Gender'] = pd.get_dummies(X_test['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        X_test['Education'] = pd.get_dummies(X_test['Education'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        
        s = '_AEG.sav'

    elif education_flag and gender_flag:
        X = df[df.columns[~df.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age'])]]
        X['Gender']= label_1.fit_transform(X['Gender'])
        X['Gender'] = pd.get_dummies(X['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        X['Education'] = pd.get_dummies(X['Education'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        X_test = df_test[df_test.columns[~df_test.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age'])]]
        X_test['Gender']= label_1.transform(X_test['Gender'])
        X_test['Gender'] = pd.get_dummies(X_test['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        X_test['Education'] = pd.get_dummies(X_test['Education'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        s = '_EG.sav'
    elif gender_flag:
        X = df[df.columns[~df.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age','Education'])]]
        X['Gender']= label_1.fit_transform(X['Gender'])
        X['Gender'] = pd.get_dummies(X['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        X_test = df_test[df_test.columns[~df_test.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age','Education'])]]
        X_test['Gender']= label_1.transform(X_test['Gender'])
        X_test['Gender'] = pd.get_dummies(X_test['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        s = '_G.sav'
    else:
        X = df[df.columns[~df.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age','Education','Gender'])]]
        X_test = df_test[df_test.columns[~df_test.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age','Education','Gender'])]]
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
    print('Classification Report for Train Set: ')
    print(classification_report(y_test, clf.predict(x_test), target_names=['E-MCI','Healthy','L-MCI','SCD']))
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
    n_scores = cross_val_score(clf, X, Y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print('Cross-Validated Accuracy : %.3f ± (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

    print('Classification Report for Test Set: ')
    print(classification_report(Y_test, clf.predict(X_test), target_names=['E-MCI','Healthy','L-MCI','SCD']))
    

    # Save the model to disk
    filename = save_model_name+s
    pickle.dump(model, open(filename, 'wb'))








## Load data sets
# df_21 = pd.read_csv('New_feat_Train_complete_some_21.csv')
# df_31 = pd.read_csv('New_feat_Train_complete_some_31.csv')
# df_41 = pd.read_csv('New_feat_Train_complete_some_41.csv')
# df_51 = pd.read_csv('New_feat_Train_complete_some_51.csv')

# df_32 = pd.read_csv('New_feat_Train_complete_some_32.csv')
# df_42 = pd.read_csv('New_feat_Train_complete_some_42.csv')
# df_52 = pd.read_csv('New_feat_Train_complete_some_52.csv')

# df_43 = pd.read_csv('New_feat_Train_complete_some_43.csv')
# df_53 = pd.read_csv('New_feat_Train_complete_some_53.csv')

# df_54 = pd.read_csv('New_feat_Train_complete_some_54.csv')

df_21 = pd.read_csv('df_some_train_21.csv')
df_31 = pd.read_csv('df_some_train_31.csv')
df_41 = pd.read_csv('df_some_train_41.csv')
df_51 = pd.read_csv('df_some_train_51.csv')

df_32 = pd.read_csv('df_some_train_32.csv')
df_42 = pd.read_csv('df_some_train_42.csv')
df_52 = pd.read_csv('df_some_train_52.csv')

df_43 = pd.read_csv('df_some_train_43.csv')
df_53 = pd.read_csv('df_some_train_53.csv')

df_54 = pd.read_csv('df_some_train_54.csv')


dfs_train = [df_21, df_31, df_41, df_51, df_32, df_42, df_52, df_43, df_53, df_54]




# df_21 = pd.read_csv('New_feat_complete_Test_21.csv')
# df_41 = pd.read_csv('New_feat_complete_Test_41.csv')
# df_31 = pd.read_csv('New_feat_complete_Test_31.csv')
# df_51 = pd.read_csv('New_feat_complete_Test_51.csv')

# df_32 = pd.read_csv('New_feat_complete_Test_32.csv')
# df_42 = pd.read_csv('New_feat_complete_Test_42.csv')
# df_52 = pd.read_csv('New_feat_complete_Test_52.csv')

# df_43 = pd.read_csv('New_feat_complete_Test_43.csv')
# df_53 = pd.read_csv('New_feat_complete_Test_53.csv')

# df_54 = pd.read_csv('New_feat_complete_Test_54.csv')

# df_21 = pd.read_csv('df_some_test_21.csv')
# df_41 = pd.read_csv('df_some_test_31.csv')
# df_31 = pd.read_csv('df_some_test_41.csv')
# df_51 = pd.read_csv('df_some_test_51.csv')

# df_32 = pd.read_csv('df_some_test_32.csv')
# df_42 = pd.read_csv('df_some_test_42.csv')
# df_52 = pd.read_csv('df_some_test_52.csv')

# df_43 = pd.read_csv('df_some_test_43.csv')
# df_53 = pd.read_csv('df_some_test_53.csv')

# df_54 = pd.read_csv('df_some_test_54.csv')

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
dfs_test = [df_21, df_31, df_41, df_51, df_32, df_42, df_52, df_43, df_53, df_54]


# model = ExtraTreesClassifier()


# models = [svm.SVC(kernel='rbf',C=10,probability=True),svm.SVC(kernel='rbf',C=20,probability=True), svm.SVC(kernel='rbf',C=12,probability=True), svm.SVC(kernel='rbf',C=14,probability=True), svm.SVC(kernel='rbf',C=14,probability=True), svm.SVC(kernel='rbf',C=14,probability=True), svm.SVC(kernel='rbf',C=14,probability=True), svm.SVC(kernel='rbf',C=10,probability=True), svm.SVC(kernel='rbf',C=25,probability=True), svm.SVC(kernel='rbf',C=10,probability=True),]
models = [ExtraTreesClassifier(),ExtraTreesClassifier(),ExtraTreesClassifier(), ExtraTreesClassifier(),ExtraTreesClassifier(),ExtraTreesClassifier(),ExtraTreesClassifier(),ExtraTreesClassifier(), ExtraTreesClassifier(),ExtraTreesClassifier()]

################ Dirty Testing!!!!
# model_names = ['SVM_21','SVM_31','SVM_41','SVM_51','SVM_32','SVM_42','SVM_52','SVM_43','SVM_53','SVM_54']
# scaler_names = ['Scaler_21.gz','Scaler_31.gz','Scaler_41.gz','Scaler_51.gz','Scaler_32.gz','Scaler_42.gz','Scaler_52.gz','Scaler_43.gz','Scaler_53.gz','Scaler_54.gz']


################ Some people out Testing:
model_names = ['SVM_21_some','SVM_31_some','SVM_41_some','SVM_51_some','SVM_32_some','SVM_42_some','SVM_52_some','SVM_43_some','SVM_53_some','SVM_54_some']
scaler_names = ['Scaler_21_some.gz','Scaler_31_some.gz','Scaler_41_some.gz','Scaler_51_some.gz','Scaler_32_some.gz','Scaler_42_some.gz','Scaler_52_some.gz','Scaler_43_some.gz','Scaler_53_some.gz','Scaler_54_some.gz']



for i in range(10):
    train_model(models[i], dfs_train[i], dfs_test[i], model_names[i], scaler_names[i], age_flag=False, education_flag=False, gender_flag=False, scaling=True)
    # train_model(models[i], dfs_train[i], dfs_test[i], model_names[i], scaler_names[i], age_flag=True, education_flag=True, gender_flag=True, scaling=True)
    # train_model(models[i], dfs_train[i], dfs_test[i], model_names[i], scaler_names[i], age_flag=True, education_flag=True, gender_flag=True, scaling=False)
    # train_model(models[i], dfs_train[i], dfs_test[i], model_names[i], scaler_names[i], age_flag=False, education_flag=False, gender_flag=False, scaling=False)
