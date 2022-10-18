from pickle import UnpicklingError
from turtle import down
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from glob import glob
import os

from zmq import XPUB_NODROP

def new_features_extraction(df_up, df_down,stage_up,stage_down, output_name):
    X_up = df_up[df_up.columns[~df_up.columns.isin(['Unnamed: 0','Name','Diagnosis','Min zcr','Age','Stress_Depression','Gender','Education'])]]
    Y_up = df_up.Diagnosis

    X_down = df_down[df_down.columns[~df_down.columns.isin(['Unnamed: 0','Name','Diagnosis','Min zcr','Age','Stress_Depression','Gender','Education'])]]
    Y_down = df_down.Diagnosis

    df_names = pd.read_csv('alz_info.csv')

    # X_down = X_down.set_index('Name')
    # X_up = X_up.set_index('Name')
    
    names = df_names['Name'].tolist()
    path = 'C:/Users/MSI User/OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης/10ο Εξάμηνο - Διπλωματική/Alz - Speech/Speech Data/Segmented Recs/'
    silence_feature_names = ['Total_Duration_Silence','# of Silences','Average Silence Duration','Median of Silence Duration','Std of Silence','Min Duration of Silence','Max Duration of Silence','Q1 Sil Duration','Q3 Sil Duration','Total non-Silent Duration','# of non-Silent','Average non-Silent Duration','Median of non-Silent Duration','Std of non-Silent Duration','Min Duration of non-Silent','Max Duration of non-Silent','Q1 non-Sil Duration','Q3 non-Sil Duration','Ratio Sil non-Sil','Ratio # Sil non-sil','Ratio Average Sil non-sil','Ratio medians','Ratio STDs','Ratio Q1','Ratio Q3']
    prosodic_feature_names = ['meanF0', 'minF0', 'maxF0', 'stdF0', 'mean_intensity', 'min_intensity', 'max_intensity', 'std_intensity', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    zcr_feature_names = ['ZeroCrossings','Max zcr','zcr']
    feature_names = silence_feature_names + prosodic_feature_names + zcr_feature_names

    features = np.empty((0,len((feature_names))),float)
    new_names = []
    for name in names:
        up_names = glob(path+name+'/*['+str(stage_up)+'].wav')
        down_names = glob(path+name+'/*['+str(stage_down)+'].wav')
        c = 0
        for up_name in up_names:
            up_name = os.path.basename(up_name)
            print(df_up['Name'])
            up_ind = df_up.index[df_up['Name'] == up_name].tolist()[0]
            row_up = X_up.iloc[up_ind].to_numpy()
            for down_name in down_names:
                down_name = os.path.basename(down_name)
                c = c+1
                # print(df_up.loc[ [up_name] , :]-df_down.loc[[down_name], :])
                # print(up_name)
                # print(down_name)
                if df_down.index[df_down['Name'] == down_name].tolist() is not np.empty:
                    down_ind = df_down.index[df_down['Name'] == down_name].tolist()[0]
                # print(df_up.iloc[up_ind])
                # print(df_down.iloc[down_ind])
                
                    row_down = X_down.iloc[down_ind].to_numpy()
                    new_row = row_up-row_down
                    features = np.append(features, [new_row], axis=0)
                    new_names.append(name+'_'+str(stage_up)+'-'+str(stage_down)+'_'+str(c))
                # print(X_up.iloc[df_up[],1:-1])

    df = pd.DataFrame(features, columns=feature_names,index=new_names)
    df.index.name = 'Name'
    df.to_csv(output_name)
    


def new_features_extraction2(df_up, df_down,stage_up,stage_down, output_name):
    X_up = df_up[df_up.columns[~df_up.columns.isin(['Unnamed: 0','Name','Diagnosis','Min zcr','Age','Stress_Depression','Gender','Education'])]]
    Y_up = df_up.Diagnosis

    X_down = df_down[df_down.columns[~df_down.columns.isin(['Unnamed: 0','Name','Diagnosis','Min zcr','Age','Stress_Depression','Gender','Education'])]]
    Y_down = df_down.Diagnosis

    df_names = pd.read_csv('alz_info.csv')

    # X_down = X_down.set_index('Name')
    # X_up = X_up.set_index('Name')
    
    names = df_names['Name'].tolist()
    path = 'C:/Users/MSI User/OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης/10ο Εξάμηνο - Διπλωματική/Alz - Speech/Speech Data/Segmented Recs/'
    silence_feature_names = ['Total_Duration_Silence','# of Silences','Average Silence Duration','Median of Silence Duration','Std of Silence','Min Duration of Silence','Max Duration of Silence','Q1 Sil Duration','Q3 Sil Duration','Total non-Silent Duration','# of non-Silent','Average non-Silent Duration','Median of non-Silent Duration','Std of non-Silent Duration','Min Duration of non-Silent','Max Duration of non-Silent','Q1 non-Sil Duration','Q3 non-Sil Duration','Ratio Sil non-Sil','Ratio # Sil non-sil','Ratio Average Sil non-sil','Ratio medians','Ratio STDs','Ratio Q1','Ratio Q3']
    prosodic_feature_names = ['meanF0', 'minF0', 'maxF0', 'stdF0', 'mean_intensity', 'min_intensity', 'max_intensity', 'std_intensity', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    zcr_feature_names = ['ZeroCrossings','Max zcr','zcr']
    feature_names = silence_feature_names + prosodic_feature_names + zcr_feature_names
    l2 = []
    features = np.empty((0,len((feature_names))),float)
    new_names = []
    for name in names:
        # up_names = glob(path+name+'/*['+str(stage_up)+'].wav')
        # down_names = glob(path+name+'/*['+str(stage_down)+'].wav')
        up_names = [n for n in df_up['Name'].tolist() if n.startswith(name)]
        down_names = [n for n in df_down['Name'].tolist() if n.startswith(name)]
        print(up_names)
        c = 0
        for up_name in up_names:
            # up_name = os.path.basename(up_name)
            # print(df_up['Name'])
            if df_up.index[df_up['Name'] == up_name].tolist() is l2:
                pass
            else:
                up_ind = df_up.index[df_up['Name'] == up_name].tolist()[0]
                row_up = X_up.iloc[up_ind].to_numpy()
                for down_name in down_names:
                    down_name = os.path.basename(down_name)
                    c = c+1
                    # print(df_up.loc[ [up_name] , :]-df_down.loc[[down_name], :])
                    # print(up_name)
                    # print(down_name)
                    if df_down.index[df_down['Name'] == down_name].tolist() is not np.empty:
                        down_ind = df_down.index[df_down['Name'] == down_name].tolist()[0]
                    # print(df_up.iloc[up_ind])
                    # print(df_down.iloc[down_ind])
                    
                        row_down = X_down.iloc[down_ind].to_numpy()
                        new_row = row_up-row_down
                        features = np.append(features, [new_row], axis=0)
                        new_names.append(name+'_'+str(stage_up)+'-'+str(stage_down)+'_'+str(c))
                    # print(X_up.iloc[df_up[],1:-1])

    df = pd.DataFrame(features, columns=feature_names,index=new_names)
    df.index.name = 'Name'
    df.to_csv(output_name)
    



# feat_per_stage = ['Complete_Features_Train_1.csv','Complete_Features_Train_2.csv','Complete_Features_Train_3.csv','Complete_Features_Train_4.csv','Complete_Features_Train_5.csv']
# df_1 = pd.read_csv(feat_per_stage[0])
# df_2 = pd.read_csv(feat_per_stage[1])
# df_3 = pd.read_csv(feat_per_stage[2])
# df_4 = pd.read_csv(feat_per_stage[3])
# df_5 = pd.read_csv(feat_per_stage[4])
# for i in df_1:
#     print(i)


# ###### NEW FEATURES EXTRACTED - 10 DIFFERENT FILES
# new_features_extraction2(df_2,df_1,2,1,'New_feat_Train_21.csv')
# new_features_extraction2(df_3,df_1,3,1,'New_feat_Train_31.csv')
# new_features_extraction2(df_4,df_1,4,1,'New_feat_Train_41.csv')
# new_features_extraction2(df_5,df_1,5,1,'New_feat_Train_51.csv')

# new_features_extraction2(df_3,df_2,3,2,'New_feat_Train_32.csv')
# new_features_extraction2(df_4,df_2,4,2,'New_feat_Train_42.csv')
# new_features_extraction2(df_5,df_2,5,2,'New_feat_Train_52.csv')

# new_features_extraction2(df_4,df_3,4,3,'New_feat_Train_43.csv')
# new_features_extraction2(df_5,df_3,5,3,'New_feat_Train_53.csv')

# new_features_extraction2(df_5,df_4,5,4,'New_feat_Train_54.csv')



# print(df_1.index[df_1['Name'] == 'Psoma_Stella_Name_1.wav'].tolist())


# feat_per_stage = ['Complete_Features_Test_1.csv','Complete_Features_Test_2.csv','Complete_Features_Test_3.csv','Complete_Features_Test_4.csv','Complete_Features_Test_5.csv']
# df_1 = pd.read_csv(feat_per_stage[0])
# df_2 = pd.read_csv(feat_per_stage[1])
# df_3 = pd.read_csv(feat_per_stage[2])
# df_4 = pd.read_csv(feat_per_stage[3])
# df_5 = pd.read_csv(feat_per_stage[4])
# ###### NEW FEATURES EXTRACTED - 10 DIFFERENT FILES
# new_features_extraction2(df_2,df_1,2,1,'New_feat_Test_21.csv')
# new_features_extraction2(df_3,df_1,3,1,'New_feat_Test_31.csv')
# new_features_extraction2(df_4,df_1,4,1,'New_feat_Test_41.csv')
# new_features_extraction2(df_5,df_1,5,1,'New_feat_Test_51.csv')

# new_features_extraction2(df_3,df_2,3,2,'New_feat_Test_32.csv')
# new_features_extraction2(df_4,df_2,4,2,'New_feat_Test_42.csv')
# new_features_extraction2(df_5,df_2,5,2,'New_feat_Test_52.csv')

# new_features_extraction2(df_4,df_3,4,3,'New_feat_Test_43.csv')
# new_features_extraction2(df_5,df_3,5,3,'New_feat_Test_53.csv')

# new_features_extraction2(df_5,df_4,5,4,'New_feat_Test_54.csv')


feat_per_stage = ['Complete_Features_some_1.csv','Complete_Features_some_2.csv','Complete_Features_some_3.csv','Complete_Features_some_4.csv','Complete_Features_some_5.csv']
df_1 = pd.read_csv(feat_per_stage[0])
df_2 = pd.read_csv(feat_per_stage[1])
df_3 = pd.read_csv(feat_per_stage[2])
df_4 = pd.read_csv(feat_per_stage[3])
df_5 = pd.read_csv(feat_per_stage[4])



###### NEW FEATURES EXTRACTED - 10 DIFFERENT FILES
new_features_extraction2(df_2,df_1,2,1,'New_feat_Train_some_21.csv')
new_features_extraction2(df_3,df_1,3,1,'New_feat_Train_some_31.csv')
new_features_extraction2(df_4,df_1,4,1,'New_feat_Train_some_41.csv')
new_features_extraction2(df_5,df_1,5,1,'New_feat_Train_some_51.csv')

new_features_extraction2(df_3,df_2,3,2,'New_feat_Train_some_32.csv')
new_features_extraction2(df_4,df_2,4,2,'New_feat_Train_some_42.csv')
new_features_extraction2(df_5,df_2,5,2,'New_feat_Train_some_52.csv')

new_features_extraction2(df_4,df_3,4,3,'New_feat_Train_some_43.csv')
new_features_extraction2(df_5,df_3,5,3,'New_feat_Train_some_53.csv')

new_features_extraction2(df_5,df_4,5,4,'New_feat_Train_some_54.csv')



most_important_feat = ['zcr', 'apq11Shimmer', 'localShimmer',
 'apq5Shimmer', 'localdbShimmer', 'meanF0', 'localabsoluteJitter',
  'ddaShimmer', 'apq3Shimmer', 'hnr', 'min_intensity',
   'Total non-Silent Duration', 'ZeroCrossings', 'Ratio # Sil non-sil',
    'Max zcr', 'stdF0', 'mean_intensity', 'max_intensity',
     'Ratio Q3', 'localJitter', 'Max Duration of non-Silent',
      'Median of Silence Duration', 'Ratio medians', 'Ratio Average Sil non-sil',
       'Ratio Sil non-Sil', 'Ratio STDs', 'Std of Silence', 
       'ddpJitter', 'Average Silence Duration', 'maxF0']