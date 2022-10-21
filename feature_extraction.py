import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import csv

from glob import glob
from pydub import AudioSegment, silence

import parselmouth
from torch import segment_reduce
from parselmouth.praat import call # type: ignore

import noisereduce as nr

import pandas as pd

### PROSODIC FEATURES - PITCH RELATED / PHONETICS FEATURES
def prosodic_features(sound, f0min, f0max, unit, interpol):
    # Unit -> "Hertz", sound -> parselmouth.Sound(voiceID), interpol -> "None", "Parabolic", etc

    ### Pitch Related
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) # Getting the Pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # Get mean pitch / Average fundamental frequency
    minF0 = call(pitch, "Get minimum", 0.0, 0.0, unit, interpol) # Get minimum pitch 
    maxF0 = call(pitch, "Get maximum", 0.0, 0.0, unit, interpol) # Get maximum pitch
    stdF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # Get standard deviation of fundamental frequency
    
    ### Intensity Related
    intensity = sound.to_intensity()
    mean_intensity = call(intensity, "Get mean", 0.0, 0.0)
    min_intensity = call(intensity, "Get minimum", 0.0, 0.0, interpol)
    max_intensity = call(intensity, "Get maximum", 0.0, 0.0, interpol)
    std_intensity = call(intensity, "Get standard deviation", 0.0, 0.0)

    ### Harmonicity Related
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0) # Harmonicity vector
    hnr = call(harmonicity, "Get mean", 0, 0) # Harmonic to Noise Ratio

    ### Prosodic Features
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    # Parameters Explained for Jitters and Shimmers -> ((Time range:) 0->, 0 (=the whole signal), shortest period=0.0001, longest period=0.02, maximum period factor=1.3, maximum amplitude factor=1.6)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    pros_feat = [meanF0, minF0, maxF0, stdF0, mean_intensity, min_intensity, max_intensity, std_intensity, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, apq5Shimmer, apq11Shimmer, ddaShimmer]

    return pros_feat

def get_prosodic_features(file_loc):
    
    unit="Hertz"
    
    filename = file_loc
    sound = parselmouth.Sound(file_loc)
    y, sr = librosa.load(file_loc)
    duration = librosa.get_duration(y=y, sr=sr)
    energy = librosa.feature.rms(y=y)
    #1
    SD_energy = np.std(energy)
    #2
    pitch = call(sound, "To Pitch", 0.0, 75, 300)

    #3
    voiced_frames = pitch.count_voiced_frames()
    total_frames = pitch.get_number_of_frames()
    #4
    voiced_to_total_ratio = voiced_frames/total_frames
    #5
    voiced_to_unvoiced_ratio =  voiced_frames / (total_frames - voiced_frames)
    
    
    return [SD_energy, voiced_frames, voiced_to_total_ratio, voiced_to_unvoiced_ratio]


## SILENCE FEATURES EXTRACTED USING pydub.silence
# Plot the signal and silences detected on it
def plot_silences(y, silences,color,title):
    plt.figure(figsize=(12,3))
    librosa.display.waveshow(y, alpha=0.5)
    for i in range(len(silences)):
        plt.plot(list(silences[i]),list([0])*len(silences[i]),color=color)
    plt.title(title)
    plt.ylim((-0.3,0.3))
    plt.show()

# Silence detection
def sil_det_normal(myaudio,long=False):
    thresh = myaudio.dBFS + myaudio.dBFS*0.5
    print("Thresh:",thresh)
    sil = silence.detect_silence(myaudio, min_silence_len=500, silence_thresh=thresh)
    sil = [((start/1000),(stop/1000)) for start,stop in sil] #convert to sec
    # print(sil)
    non_sil = silence.detect_nonsilent(myaudio,min_silence_len=500, silence_thresh=thresh)
    non_sil = [((start/1000),(stop/1000)) for start,stop in non_sil] #convert to sec
    
    # non_sil = []
    # for i in range(1,len(sil)):
    #     non_sil.append(tuple([list(sil[i-1])[1],list(sil[i])[0]]))
    # # print(non_sil)
    return sil,non_sil


# Features for silences
def sil_features(silences, non_sil):
    silence_features = []
    # Finding durations for each silence
    durations_of_silence = [list(silences[i])[1]-list(silences[i])[0] for i in range(len(silences))]
    # # # print(durations_of_silence)
    # Finding the total duration of silences
    sum_durations_sil = sum(durations_of_silence)
    silence_features.append(sum_durations_sil)
    # # # print(sum_durations_sil)
    # Number of silences
    no_of_silences = len(durations_of_silence)
    silence_features.append(no_of_silences)
    # # # print(no_of_silences)
    # Average silence duration
    if(no_of_silences>0):
        average_silence_duration = sum_durations_sil/no_of_silences
        silence_features.append(average_silence_duration)
        # Median of the silence durations
        med_sil = np.median(durations_of_silence)
        silence_features.append(med_sil)
        # Standard deviation of silence duration
        std_sil = np.std(durations_of_silence)
        silence_features.append(std_sil)
        # Min Max
        silence_features.append(np.min(durations_of_silence))
        silence_features.append(np.max(durations_of_silence))
        # Q1-Q3 Quartiles
        Q1_sil = np.percentile(durations_of_silence, 25)
        Q3_sil = np.percentile(durations_of_silence, 75)
        silence_features.append(Q1_sil)
        silence_features.append(Q3_sil)
    else:
        average_silence_duration = 0
        silence_features.append(average_silence_duration)
        # Median of the silence durations
        med_sil = 0
        silence_features.append(med_sil)
        # Standard deviation of silence duration
        std_sil = 0
        silence_features.append(std_sil)
        # Min Max
        silence_features.append(0)
        silence_features.append(0)
        # Q1-Q3 Quartiles
        Q1_sil = 0
        Q3_sil = 0
        silence_features.append(Q1_sil)
        silence_features.append(Q3_sil)

    # Finding durations for non silent
    durations_of_non_sil = [list(non_sil[i])[1]-list(non_sil[i])[0] for i in range(len(non_sil))]
    # Sum of durations of non silent regions
    sum_durations_non_sil = sum(durations_of_non_sil)
    silence_features.append(sum_durations_non_sil)
    # Number of non silent regions
    no_of_non_sil = len(durations_of_non_sil)
    silence_features.append(no_of_non_sil)
    # Average non-silent duration - Mean
    average_non_sil_duration = sum_durations_non_sil/no_of_non_sil
    silence_features.append(average_non_sil_duration)
    # Median of the non-silent durations
    med_non_sil = np.median(durations_of_non_sil)
    silence_features.append(med_non_sil)
    # Standard deviation of non silenct duration
    std_non_sil = np.std(durations_of_non_sil)
    silence_features.append(std_non_sil)
    # Min Max
    silence_features.append(np.min(durations_of_non_sil))
    silence_features.append(np.max(durations_of_non_sil))
    # Q1-Q3 Quartiles
    Q1_non_sil = np.percentile(durations_of_non_sil, 25)
    Q3_non_sil = np.percentile(durations_of_non_sil, 75)
    silence_features.append(Q1_non_sil)
    silence_features.append(Q3_non_sil)

    # Ratio of silent vs non silent durations
    ratio_sil_non_sil = sum_durations_sil/sum_durations_non_sil
    silence_features.append(ratio_sil_non_sil)
    # Ratio of number of silent vs non silent regions
    ratio_sil_non_sil_no = no_of_silences/no_of_non_sil
    silence_features.append(ratio_sil_non_sil_no)
    # Ratio of average_silence_duration vs average_non_sil_duration - Mean
    ratio_average_sil_non_sil = average_silence_duration/average_non_sil_duration
    silence_features.append(ratio_average_sil_non_sil)
    # Ratio of medians
    ratio_med = med_sil/med_non_sil
    silence_features.append(ratio_med)
    # Ratio of std
    ratio_std = std_sil/std_non_sil
    silence_features.append(ratio_std)
    # Ratio of Q1
    ratio_Q1 = Q1_sil/Q1_non_sil
    silence_features.append(ratio_Q1)
    ratio_Q3 = Q3_sil/Q3_non_sil
    silence_features.append(ratio_Q3)

    return silence_features

### ZERO CROSSING FEATURES
def zero_crossing_features(sound):
    # Zero crossings
    zc = librosa.zero_crossings(sound, pad=False)
    zc = sum(zc)
    # ZCR
    dur = librosa.get_duration(sound)
    zcr = zc/dur
    zcr = librosa.feature.zero_crossing_rate(sound)

    ## ZCR lowerst and highest instantaneous value
    min_zcr = zcr.min()
    max_zcr = zcr.max()

    ## ZCR mean
    zcr = zcr.mean()

    return [zc,min_zcr,max_zcr,zcr]


def feature_extraction_per_stage(files_No, output_name, going_for_all=False):
    ## going_for_all -> bool to say if we want feature extraction for a new file (TRUE)
    ##                  or if we want to use the function for a new person and thus
    ##                  only write the new features on an existing csv file.

    # Feature names and feature array intialization    
    names = []
    silence_feature_names = ['Total_Duration_Silence','# of Silences','Average Silence Duration','Median of Silence Duration','Std of Silence','Min Duration of Silence','Max Duration of Silence','Q1 Sil Duration','Q3 Sil Duration','Total non-Silent Duration','# of non-Silent','Average non-Silent Duration','Median of non-Silent Duration','Std of non-Silent Duration','Min Duration of non-Silent','Max Duration of non-Silent','Q1 non-Sil Duration','Q3 non-Sil Duration','Ratio Sil non-Sil','Ratio # Sil non-sil','Ratio Average Sil non-sil','Ratio medians','Ratio STDs','Ratio Q1','Ratio Q3']
    prosodic_feature_names = ['meanF0', 'minF0', 'maxF0', 'stdF0', 'mean_intensity', 'min_intensity', 'max_intensity', 'std_intensity', 'hnr', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    zcr_feature_names = ['ZeroCrossings','Min zcr','Max zcr','zcr']
    feature_names = silence_feature_names + prosodic_feature_names + zcr_feature_names
    features = np.empty((0,len((feature_names))),float)
    print(len(files_No)," files are expected to go through feature extraction.")
    i = 1
    # Basic loop
    for file in files_No:
        temp_features = np.array([])
        names.append(os.path.basename(file))
        snd = parselmouth.Sound(file)
        myaudio = AudioSegment.from_wav(file)
        y, sr = librosa.load(file)

        # Extracting silence features
        silences,non_silent = sil_det_normal(myaudio)
        # plot_silences(y, silences,'r',os.path.basename(file))
        silence_features = sil_features(silences,non_silent)
        # print("Number of silence features extracted: ", len(silence_features))
        temp_features = np.append(temp_features,silence_features)
        
        # Prosodic features
        pros_features = prosodic_features(snd, 75, 500.0, "Hertz", "parabolic")
        temp_features = np.append(temp_features,pros_features)
        # print("Number of prosodic features extracted: ", len(pros_features))

        # Zero-crossing features
        zcr_features = zero_crossing_features(y)
        temp_features = np.append(temp_features,zcr_features)
        # print("Number of zcr features extracted: ", len(zcr_features))

        # print("Total Number of features extracted: ", len(features))
        print(i/len(files_No)*100," % ","of Feature extraction Completed. File : ", i, " out of ", len(files_No))
        i = i+1
        
        features = np.append(features, [temp_features], axis=0)

    ## Writing a new file    
    if going_for_all:
        df = pd.DataFrame(features, columns=feature_names,index=names)
        df.to_csv(output_name)
    ## Writing on existing file (when adding new people to the database)
    else:
        df = pd.DataFrame(features,columns=feature_names, index=names)
        df.to_csv(output_name, mode='a', index=True, header=False)

    

### Function to extract and save features for new people to the database   
def feature_extraction_new_person(persons_folder,csv_output_names):
    # Getting different bunches of files according to stage of the recording
    files1 = glob(persons_folder+'/*[1].wav')
    files2 = glob(persons_folder+'/*[2].wav')
    files3 = glob(persons_folder+'/*[3].wav')
    files4 = glob(persons_folder+'/*[4].wav')
    files5 = glob(persons_folder+'/*[5].wav')
    files = [files1, files2, files3, files4, files5]
    for i in range(5):
        print("Stage ", i+1, " :")
        feature_extraction_per_stage(files[i],csv_output_names[i], going_for_all=False)
        
    

    




seg_recs_path = 'C:/Users/MSI User/OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης/10ο Εξάμηνο - Διπλωματική/Alz - Speech/Speech Data/Segmented Recs/*/'

# Files stored per stage of recording
files1 = glob(seg_recs_path+'/*[1].wav')
files2 = glob(seg_recs_path+'/*[2].wav')
files3 = glob(seg_recs_path+'/*[3].wav')
files4 = glob(seg_recs_path+'/*[4].wav')
files5 = glob(seg_recs_path+'/*[5].wav')


# feature_extraction_per_stage(files2, 'Features_Stage_2.csv',going_for_all=True)
folder = 'C:/Users/MSI User/OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης/10ο Εξάμηνο - Διπλωματική/Alz - Speech/Speech Data/Segmented Recs/name'



csv_names = ['test.csv','test2.csv','test3.csv','test4.csv','test5.csv']
feature_extraction_new_person(folder,csv_names)
