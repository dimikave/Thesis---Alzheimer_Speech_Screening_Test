from re import M
import numpy as np
import pandas as pd






def five_recs_feat(dfs, output_name):

    df_names = pd.read_csv('alz_info.csv')
    
    names = df_names['Name'].tolist()
    path = 'C:/Users/MSI User/OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης/10ο Εξάμηνο - Διπλωματική/Alz - Speech/Speech Data/Segmented Recs/'
    silence_feature_names = ['Total_Duration_Silence','# of Silences','Average Silence Duration','Median of Silence Duration','Std of Silence','Min Duration of Silence','Max Duration of Silence','Q1 Sil Duration','Q3 Sil Duration','Total non-Silent Duration','# of non-Silent','Average non-Silent Duration','Median of non-Silent Duration','Std of non-Silent Duration','Min Duration of non-Silent','Max Duration of non-Silent','Q1 non-Sil Duration','Q3 non-Sil Duration','Ratio Sil non-Sil','Ratio # Sil non-sil','Ratio Average Sil non-sil','Ratio medians','Ratio STDs','Ratio Q1','Ratio Q3']
    prosodic_feature_names = ['meanF0', 'minF0', 'maxF0', 'stdF0', 'mean_intensity', 'min_intensity', 'max_intensity', 'std_intensity', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    zcr_feature_names = ['ZeroCrossings','Min zcr','Max zcr','zcr']
    all_names = silence_feature_names + prosodic_feature_names + zcr_feature_names
    feature_names = []

    for i in range(1,6):
        temp_names = [s+'_'+str(i) for s in all_names]
        feature_names.extend(temp_names)
        # dfs[i-1] = dfs[i-1][dfs[i-1].columns[~dfs[i-1].columns.isin(['Unnamed: 0','Ratio Q1', 'Name','Diagnosis','Min zcr'])]]
    
    features = np.empty((0,len((feature_names))),float)
    print(len(feature_names))
    new_names = []
    for name in names:
        c = 0
        temp_features = np.array([])
        temp_features1 = np.array([])
        temp_features2 = np.array([])
        temp_features3 = np.array([])
        temp_features4 = np.array([])
        print(name)
        for i, row in dfs[0].iterrows():
            if row.tolist()[0].startswith(name):
                temp_features = np.append(temp_features,row.tolist()[1:])
                for j,row1 in dfs[1].iterrows():
                    if row1.tolist()[0].startswith(name):
                        temp_features1 = np.append(temp_features,row1.tolist()[1:])
                        for k, row2 in dfs[2].iterrows():
                            if row2.tolist()[0].startswith(name):
                                temp_features2 = np.append(temp_features1,row1.tolist()[1:])
                                for l, row3 in dfs[3].iterrows():
                                    if row3.tolist()[0].startswith(name):
                                        temp_features3 = np.append(temp_features2,row3.tolist()[1:])
                                        for m, row4 in dfs[4].iterrows():
                                            if row4.tolist()[0].startswith(name):
                                                temp_features4 = np.append(temp_features3,row4.tolist()[1:])
                                                # print(len(temp_features))
                                                features = np.append(features, [temp_features4], axis=0)
                                                # print(features.shape)
                                                c = c+1
                                                new_names.append(name + '_'+str(c))
                                                # print(new_names[-1])
                                            temp_features4 = np.array([])
                                    temp_features3 = np.array([])
                            temp_features2 = np.array([])
                    temp_features1 = np.array([])
            temp_features = np.array([])
    
    df = pd.DataFrame(features, columns=feature_names,index=new_names)
    df.to_csv(output_name)









feat_per_stage = ['Features_Stage_1.csv','Features_Stage_2.csv','Features_Stage_3.csv','Features_Stage_4.csv','Features_Stage_5.csv']
df_1 = pd.read_csv(feat_per_stage[0])
df_2 = pd.read_csv(feat_per_stage[1])
df_3 = pd.read_csv(feat_per_stage[2])
df_4 = pd.read_csv(feat_per_stage[3])
df_5 = pd.read_csv(feat_per_stage[4])
dfs = [df_1, df_2, df_3, df_4, df_5]
# dfs[0].set_index("Name", 
#               inplace = True)
# print(dfs[0].loc[['Salpistis_Dimitrios_Name_1.wav']].values.flatten().tolist())
# for i, row in dfs[0].iterrows():
#     print(i)
#     print(row.tolist()[1:])

five_recs_feat(dfs,'Final_features_penta.csv')