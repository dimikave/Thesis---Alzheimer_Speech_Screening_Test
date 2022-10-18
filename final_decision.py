import tempfile
from tkinter.tix import Y_REGION
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

warnings.filterwarnings('ignore')

import time






def final_model_predict(models,scalers, dfs_person, age_flag, education_flag, gender_flag, scaling):
    
    preds = np.empty((1,4),float)
    
    for i in range(10):
        df = dfs_person[i]
        # print(df.head())
        # print(df.head())

        ### Fill NaN and Inf
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        
        ### Prepare the data-set
        Y = df['Diagnosis']
        label_1 = LabelEncoder()

        if age_flag and education_flag and gender_flag:
            X = df[df.columns[~df.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr'])]]
            X['Gender']= label_1.fit_transform(X['Gender'])
            X['Gender'] = pd.get_dummies(X['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
            X['Education'] = pd.get_dummies(X['Education'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)

        elif education_flag and gender_flag:
            X = df[df.columns[~df.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age'])]]
            X['Gender']= label_1.fit_transform(X['Gender'])
            X['Gender'] = pd.get_dummies(X['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
            X['Education'] = pd.get_dummies(X['Education'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)

        elif gender_flag:
            X = df[df.columns[~df.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age','Education'])]]
            X['Gender']= label_1.fit_transform(X['Gender'])
            X['Gender'] = pd.get_dummies(X['Gender'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)

        else:
            X = df[df.columns[~df.columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr','Age','Education','Gender'])]]

        ### Encoding
        X['Stress_Depression']= label_1.fit_transform(X['Stress_Depression'])
        X['Stress_Depression'] = pd.get_dummies(X['Stress_Depression'],prefix_sep='_', dummy_na=False, columns=None,sparse=False, drop_first=False)
        ## Scaling if need to
        if scaling:
            scaler = joblib.load(scalers[i])
            # x_train = scaler.fit_transform(x_train)
            # x_test = scaler.transform(x_test)
            X = scaler.transform(X)
        X = pd.DataFrame(X)
        # print(X)
        ### Load Model
        loaded_model = pickle.load(open(models[i], 'rb'))
        # print(loaded_model.predict_proba(X))
        ### Predict
        preds = np.append(preds, loaded_model.predict_proba(X), axis=0)
        print(loaded_model.predict(X))

    preds = np.sum(preds, axis=0)
    label = np.argmax(preds)
    classes = loaded_model.classes_
    # print(preds)
    label = classes[label]
    
    return label





# Load data sets
df_21 = pd.read_csv('New_feat_complete_Test_21.csv')
df_41 = pd.read_csv('New_feat_complete_Test_41.csv')
df_31 = pd.read_csv('New_feat_complete_Test_31.csv')
df_51 = pd.read_csv('New_feat_complete_Test_51.csv')

df_32 = pd.read_csv('New_feat_complete_Test_32.csv')
df_42 = pd.read_csv('New_feat_complete_Test_42.csv')
df_52 = pd.read_csv('New_feat_complete_Test_52.csv')

df_43 = pd.read_csv('New_feat_complete_Test_43.csv')
df_53 = pd.read_csv('New_feat_complete_Test_53.csv')

df_54 = pd.read_csv('New_feat_complete_Test_54.csv')

# df_21 = pd.read_csv('New_feat_complete_21.csv')
# df_41 = pd.read_csv('New_feat_complete_41.csv')
# df_31 = pd.read_csv('New_feat_complete_31.csv')
# df_51 = pd.read_csv('New_feat_complete_51.csv')

# df_32 = pd.read_csv('New_feat_complete_32.csv')
# df_42 = pd.read_csv('New_feat_complete_42.csv')
# df_52 = pd.read_csv('New_feat_complete_52.csv')

# df_43 = pd.read_csv('New_feat_complete_43.csv')
# df_53 = pd.read_csv('New_feat_complete_53.csv')

# df_54 = pd.read_csv('New_feat_complete_54.csv')
dfs = [df_21, df_31, df_41, df_51, df_32, df_42, df_52, df_43, df_53, df_54]


models = ['SVM_21.sav','SVM_31.sav','SVM_41.sav','SVM_51.sav','SVM_32.sav','SVM_42.sav','SVM_52.sav','SVM_43.sav','SVM_53.sav','SVM_54.sav']
scalers = ['Scaler_21.gz','Scaler_31.gz','Scaler_41.gz','Scaler_51.gz','Scaler_32.gz','Scaler_42.gz','Scaler_52.gz','Scaler_43.gz','Scaler_53.gz','Scaler_54.gz']

# models = ['SVM_21_some.sav','SVM_31_some.sav','SVM_41_some.sav','SVM_51_some.sav','SVM_32_some.sav','SVM_42_some.sav','SVM_52_some.sav','SVM_43_some.sav','SVM_53_some.sav','SVM_54_some.sav']
# scalers = ['Scaler_21_some.gz','Scaler_31_some.gz','Scaler_41_some.gz','Scaler_51_some.gz','Scaler_32_some.gz','Scaler_42_some.gz','Scaler_52_some.gz','Scaler_43_some.gz','Scaler_53_some.gz','Scaler_54_some.gz']

# models = ['SVM_21_some_AEG.sav','SVM_31_some_AEG.sav','SVM_41_some_AEG.sav','SVM_51_some_AEG.sav','SVM_32_some_AEG.sav','SVM_42_some_AEG.sav','SVM_52_some_AEG.sav','SVM_43_some_AEG.sav','SVM_53_some_AEG.sav','SVM_54_some_AEG.sav']
# scalers = ['Scaler_21_some.gz','Scaler_31_some.gz','Scaler_41_some.gz','Scaler_51_some.gz','Scaler_32_some.gz','Scaler_42_some.gz','Scaler_52_some.gz','Scaler_43_some.gz','Scaler_53_some.gz','Scaler_54_some.gz']




#### FINAL TESTING - Dirty or left out ##########
Y_real = []
Y_predictions = []
dfs_person = []

# df_penta = pd.read_csv('Final_Features_penta_complete.csv')
df_penta = pd.read_csv('Final_Features_penta_complete_some_test.csv')

print(df_penta.Diagnosis.value_counts())

silence_feature_names = ['Total_Duration_Silence','# of Silences','Average Silence Duration','Median of Silence Duration','Std of Silence','Min Duration of Silence','Max Duration of Silence','Q1 Sil Duration','Q3 Sil Duration','Total non-Silent Duration','# of non-Silent','Average non-Silent Duration','Median of non-Silent Duration','Std of non-Silent Duration','Min Duration of non-Silent','Max Duration of non-Silent','Q1 non-Sil Duration','Q3 non-Sil Duration','Ratio Sil non-Sil','Ratio # Sil non-sil','Ratio Average Sil non-sil','Ratio medians','Ratio STDs','Ratio Q1','Ratio Q3']
prosodic_feature_names = ['meanF0', 'minF0', 'maxF0', 'stdF0', 'mean_intensity', 'min_intensity', 'max_intensity', 'std_intensity', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
zcr_feature_names = ['ZeroCrossings','Min zcr','Max zcr','zcr']
feature_names = silence_feature_names + prosodic_feature_names + zcr_feature_names + ['Gender','Age','Education','Stress_Depression','Diagnosis']


# feature_names = [s + '_1' for s in feature_names]
t = time.time()

for i,row in df_penta.iterrows():
    print(i)
    # print(df_penta.shape)
    dfs_person = []
    stage1 = df_penta.iloc[[i],0*49+2:0*49+49+2]
    stage2 = df_penta.iloc[[i],1*49+2:1*49+49+2]
    stage3 = df_penta.iloc[[i],2*49+2:2*49+49+2]
    stage4 = df_penta.iloc[[i],3*49+2:3*49+49+2]
    stage5 = df_penta.iloc[[i],4*49+2:4*49+49+2]

    stage1 = stage1.values.tolist()
    stage2 = stage2.values.tolist()
    stage3 = stage3.values.tolist()
    stage4 = stage4.values.tolist()
    stage5 = stage5.values.tolist()

    diff21 = np.subtract(stage2,stage1)
    diff31 = np.subtract(stage3,stage1)
    diff41 = np.subtract(stage4,stage1)
    diff51 = np.subtract(stage5,stage1)
    diff32 = np.subtract(stage3,stage2)
    diff42 = np.subtract(stage4,stage2)
    diff52 = np.subtract(stage5,stage2)
    diff43 = np.subtract(stage4,stage3)
    diff53 = np.subtract(stage5,stage3)
    diff54 = np.subtract(stage5,stage4)

    completion = [row.Gender, row.Age, row.Education, row.Stress_Depression, row.Diagnosis]
    diff21 = np.append(diff21,completion)
    diff31 = np.append(diff31,completion)
    diff41 = np.append(diff41,completion)
    diff51 = np.append(diff51,completion)
    diff32 = np.append(diff32,completion)
    diff42 = np.append(diff42,completion)
    diff52 = np.append(diff52,completion)
    diff43 = np.append(diff43,completion)
    diff53 = np.append(diff53,completion)
    diff54 = np.append(diff54,completion)

    df_n_21 = pd.DataFrame([diff21],columns=feature_names)
    dfs_person.append(df_n_21)
    df_n_31 = pd.DataFrame([diff31],columns=feature_names)
    dfs_person.append(df_n_31)
    df_n_41 = pd.DataFrame([diff41],columns=feature_names)
    dfs_person.append(df_n_41)
    df_n_51 = pd.DataFrame([diff51],columns=feature_names)
    dfs_person.append(df_n_51)
    df_n_32 = pd.DataFrame([diff32],columns=feature_names)
    dfs_person.append(df_n_32)
    df_n_42 = pd.DataFrame([diff42],columns=feature_names)
    dfs_person.append(df_n_42)
    df_n_52 = pd.DataFrame([diff52],columns=feature_names)
    dfs_person.append(df_n_52)
    df_n_43 = pd.DataFrame([diff43],columns=feature_names)
    dfs_person.append(df_n_43)
    df_n_53 = pd.DataFrame([diff53],columns=feature_names)
    dfs_person.append(df_n_53)
    df_n_54 = pd.DataFrame([diff54],columns=feature_names)
    dfs_person.append(df_n_54)

    # if df_n_21.isnull().values.any() or df_n_31.isnull().values.any() or df_n_41.isnull().values.any() or df_n_51.isnull().values.any() or df_n_32.isnull().values.any() or df_n_42.isnull().values.any() or df_n_52.isnull().values.any() or df_n_43.isnull().values.any() or df_n_53.isnull().values.any() or df_n_54.isnull().values.any():
    #     continue
    # if ~np.isinfinity().values.any() or df_n_31.isnull().values.any() or df_n_41.isnull().values.any() or df_n_51.isnull().values.any() or df_n_32.isnull().values.any() or df_n_42.isnull().values.any() or df_n_52.isnull().values.any() or df_n_43.isnull().values.any() or df_n_53.isnull().values.any() or df_n_54.isnull().values.any():
    #     continue
    # print(dfs_person)
    Y_pred = final_model_predict(models,scalers, dfs_person,False,False,False,True)
    # Y_pred = final_model_predict(models,scalers, dfs_person,True,True,True,True)
    Y_predictions.append(Y_pred)
    # print(Y_pred,' vs ', row.Diagnosis)
    Y_real.append(row.Diagnosis)
    print(Y_predictions[-1],' vs ', Y_real[-1])
    
    
    
elapsed = time.time() - t
print(elapsed)

df_disp = pd.DataFrame({'True Diagnosis': Y_real,'Prediction Diagnosis': Y_predictions})
df_disp.to_csv('Dirty_test_results.csv')
# print(df_disp.head(45))

ax = plt.subplot()
cm = confusion_matrix(Y_real,Y_predictions)
sns.heatmap(cm,annot=True,fmt="d",ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['E-MCI','Healthy','L-MCI','SCD'])
ax.yaxis.set_ticklabels(['E-MCI','Healthy','L-MCI','SCD'])
# print(Y_real.value_counts())
plt.show()
print(classification_report(Y_real,Y_predictions))


