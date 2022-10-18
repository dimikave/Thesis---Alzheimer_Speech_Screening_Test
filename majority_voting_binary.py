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
    
    preds = np.empty((1,2),float)
    
    for i in range(5):
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
        # print(loaded_model.predict(X))

    preds = np.sum(preds, axis=0)
    print(preds)
    label = np.argmax(preds)
    classes = loaded_model.classes_
    # print(preds)
    label = classes[label]
    
    return label





# Load data sets
df1_test = pd.read_csv('df_some_test_1.csv').dropna().reset_index(drop = True)
df2_test = pd.read_csv('df_some_test_2.csv').dropna().reset_index(drop = True)
df3_test = pd.read_csv('df_some_test_3.csv').dropna().reset_index(drop = True)
df4_test = pd.read_csv('df_some_test_4.csv').dropna().reset_index(drop = True)
df5_test = pd.read_csv('df_some_test_5.csv').dropna().reset_index(drop = True)


dfs = [df1_test, df2_test, df3_test, df4_test, df5_test]


#### USE train_final_models.py TO EXTRACT MODELS/SCALERS
# models = ['SVM_1_hs.sav','SVM_2_hs.sav','SVM_3_hs.sav','SVM_4_hs.sav','SVM_5_hs.sav']
# scalers = ['Scaler_1_hs.gz','Scaler_2_hs.gz','Scaler_3_hs.gz','Scaler_4_hs.gz','Scaler_5_hs.gz']

models = ['SVM_1_hm.sav','SVM_2_hm.sav','SVM_3_hm.sav','SVM_4_hm.sav','SVM_5_hm.sav']
scalers = ['Scaler_1_hm.gz','Scaler_2_hs.gz','Scaler_3_hm.gz','Scaler_4_hm.gz','Scaler_5_hm.gz']

# models = ['SVM_1_sm.sav','SVM_2_sm.sav','SVM_3_sm.sav','SVM_4_sm.sav','SVM_5_sm.sav']
# scalers = ['Scaler_1_sm.gz','Scaler_2_sm.gz','Scaler_3_sm.gz','Scaler_4_sm.gz','Scaler_5_sm.gz']



#### FINAL TESTING - Dirty or left out ##########
Y_real = []
Y_predictions = []
dfs_person = []

# df_penta = pd.read_csv('Final_Features_penta_complete.csv')
# df_penta = pd.read_csv('Final_Features_penta_complete_some_test.csv')
# df_penta = pd.read_csv('Final_Features_penta_complete_some_test_hs.csv')
df_penta = pd.read_csv('Final_Features_penta_complete_some_test_hm.csv')
# df_penta = pd.read_csv('Final_Features_penta_complete_some_test_sm.csv')

df_penta.reset_index()

print(df_penta.Diagnosis.value_counts())

silence_feature_names = ['Total_Duration_Silence','# of Silences','Average Silence Duration','Median of Silence Duration','Std of Silence','Min Duration of Silence','Max Duration of Silence','Q1 Sil Duration','Q3 Sil Duration','Total non-Silent Duration','# of non-Silent','Average non-Silent Duration','Median of non-Silent Duration','Std of non-Silent Duration','Min Duration of non-Silent','Max Duration of non-Silent','Q1 non-Sil Duration','Q3 non-Sil Duration','Ratio Sil non-Sil','Ratio # Sil non-sil','Ratio Average Sil non-sil','Ratio medians','Ratio STDs','Ratio Q1','Ratio Q3']
prosodic_feature_names = ['meanF0', 'minF0', 'maxF0', 'stdF0', 'mean_intensity', 'min_intensity', 'max_intensity', 'std_intensity', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
zcr_feature_names = ['ZeroCrossings','Min zcr','Max zcr','zcr']
feature_names = silence_feature_names + prosodic_feature_names + zcr_feature_names + ['Gender','Age','Education','Stress_Depression','Diagnosis']


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

    completion = [row.Gender, row.Age, row.Education, row.Stress_Depression, row.Diagnosis]
    stage1 = np.append(stage1,completion)
    stage2 = np.append(stage2,completion)
    stage3 = np.append(stage3,completion)
    stage4 = np.append(stage4,completion)
    stage5 = np.append(stage5,completion)

    stage_n_1 = pd.DataFrame([stage1],columns=feature_names)
    stage_n_2 = pd.DataFrame([stage2],columns=feature_names)
    stage_n_3 = pd.DataFrame([stage3],columns=feature_names)
    stage_n_4 = pd.DataFrame([stage4],columns=feature_names)
    stage_n_5 = pd.DataFrame([stage5],columns=feature_names)

    
    dfs_person.append(stage_n_1)
    dfs_person.append(stage_n_2)
    dfs_person.append(stage_n_3)
    dfs_person.append(stage_n_4)
    dfs_person.append(stage_n_5)

    Y_pred = final_model_predict(models,scalers, dfs_person,False,False,False,True)
    # Y_pred = final_model_predict(models,scalers, dfs_person,True,True,True,True)
    Y_predictions.append(Y_pred)
    # print(Y_pred,' vs ', row.Diagnosis)
    Y_real.append(row.Diagnosis)
    print(Y_predictions[-1],' vs ', Y_real[-1])
    
    
    
elapsed = time.time() - t
print(elapsed)

df_disp = pd.DataFrame({'True Diagnosis': Y_real,'Prediction Diagnosis': Y_predictions})
df_disp.to_csv('Binary_mv_hm.csv')
# print(df_disp.head(45))

ax = plt.subplot()
cm = confusion_matrix(Y_real,Y_predictions)
sns.heatmap(cm,annot=True,fmt="d",ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Healthy','MCI'])
ax.yaxis.set_ticklabels(['Healthy','MCI'])
# print(Y_real.value_counts())
plt.show()
print(classification_report(Y_real,Y_predictions))


