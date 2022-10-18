import pandas as pd
import numpy as np
import os
from glob import glob


def complete_csv(old_csvs, new_csv):
    dfInfo = pd.read_csv('alz_info.csv')
    names = dfInfo["Name"].tolist()
    for i in range(len(old_csvs)):
        df = pd.read_csv(old_csvs[i])
        filenames = df["Name"].tolist()
        for name in names:
            print(name)
            for filename in filenames:
                if filename.startswith(name):
                    gender = dfInfo.loc[dfInfo.Name==name,"Gender"].tolist()[0]
                    age = dfInfo.loc[dfInfo.Name==name,"Age"].tolist()[0]
                    edu = dfInfo.loc[dfInfo.Name==name,"Education"].tolist()[0]
                    stress = dfInfo.loc[dfInfo.Name==name,"Stress_Depression"].tolist()[0]
                    diagnosis = dfInfo.loc[dfInfo.Name==name,"Diagnosis"].tolist()[0]
                    df.loc[df.Name==filename,"Gender"] = gender
                    df.loc[df.Name==filename,"Age"] = age
                    df.loc[df.Name==filename,"Education"] = edu
                    df.loc[df.Name==filename,"Stress_Depression"] = stress
                    df.loc[df.Name==filename,"Diagnosis"] = diagnosis
                    
        df.to_csv(new_csv[i])


# old = ['Features_Stage_1.csv','Features_Stage_2.csv','Features_Stage_3.csv','Features_Stage_4.csv','Features_Stage_5.csv']
# new = ['Complete_Features_1.csv','Complete_Features_2.csv','Complete_Features_3.csv','Complete_Features_4.csv','Complete_Features_5.csv']

# old = ['New_feat_21.csv','New_feat_31.csv','New_feat_41.csv','New_feat_51.csv','New_feat_32.csv','New_feat_42.csv','New_feat_52.csv','New_feat_43.csv','New_feat_53.csv','New_feat_54.csv']
# new = ['New_feat_complete_21.csv','New_feat_complete_31.csv','New_feat_complete_41.csv','New_feat_complete_51.csv','New_feat_complete_32.csv','New_feat_complete_42.csv','New_feat_complete_52.csv','New_feat_complete_43.csv','New_feat_complete_53.csv','New_feat_complete_54.csv']

# old = ['Final_features_penta.csv']
# new = ['Final_Features_penta_complete.csv']


# old = ['New_feat_Train_21.csv','New_feat_Train_31.csv','New_feat_Train_41.csv','New_feat_Train_51.csv','New_feat_Train_32.csv','New_feat_Train_42.csv','New_feat_Train_52.csv','New_feat_Train_43.csv','New_feat_Train_53.csv','New_feat_Train_54.csv']
# new = ['New_feat_complete_Train_21.csv','New_feat_complete_Train_31.csv','New_feat_complete_Train_41.csv','New_feat_complete_Train_51.csv','New_feat_complete_Train_32.csv','New_feat_complete_Train_42.csv','New_feat_complete_Train_52.csv','New_feat_complete_Train_43.csv','New_feat_complete_Train_53.csv','New_feat_complete_Train_54.csv']

# complete_csv(old,new)

# old = ['New_feat_Test_21.csv','New_feat_Test_31.csv','New_feat_Test_41.csv','New_feat_Test_51.csv','New_feat_Test_32.csv','New_feat_Test_42.csv','New_feat_Test_52.csv','New_feat_Test_43.csv','New_feat_Test_53.csv','New_feat_Test_54.csv']
# new = ['New_feat_complete_Test_21.csv','New_feat_complete_Test_31.csv','New_feat_complete_Test_41.csv','New_feat_complete_Test_51.csv','New_feat_complete_Test_32.csv','New_feat_complete_Test_42.csv','New_feat_complete_Test_52.csv','New_feat_complete_Test_43.csv','New_feat_complete_Test_53.csv','New_feat_complete_Test_54.csv']



old = ['New_feat_Train_some_21.csv','New_feat_Train_some_31.csv','New_feat_Train_some_41.csv','New_feat_Train_some_51.csv','New_feat_Train_some_32.csv','New_feat_Train_some_42.csv','New_feat_Train_some_52.csv','New_feat_Train_some_43.csv','New_feat_Train_some_53.csv','New_feat_Train_some_54.csv']
new = ['New_feat_Train_complete_some_21.csv','New_feat_Train_complete_some_31.csv','New_feat_Train_complete_some_41.csv','New_feat_Train_complete_some_51.csv','New_feat_Train_complete_some_32.csv','New_feat_Train_complete_some_42.csv','New_feat_Train_complete_some_52.csv','New_feat_Train_complete_some_43.csv','New_feat_Train_complete_some_53.csv','New_feat_Train_complete_some_54.csv']

complete_csv(old,new)
