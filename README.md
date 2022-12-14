# Cognitive decline detection using speech features: A machine learning approach

## Abstract
The ever-increasing impact of neurocognitive diseases is more and more apparent, as
statistics show that due to the longer life expectancy established today, minor (Mild
Cognitive Impairment-MCI) and major cognitive diseases (Dementia) will soon be a
societal problem that cannot be ignored. Most of the currently established methods
of neurodegeneration diagnosis are either invasive (blood tests, neuroimaging) and/or
require a full neuropsychological and clinical assessment, which is performed in a clinical
environment and usually requires a lot of time. To make the diagnosis process simpler,
studies exist that focus on the speech decline which usually accompanies the cognitive
one, so as to classify people according to their cognitive status, often by collecting speech
data from structured interviews and deploying a machine learning model.

In this study, the validity of a multiclass classification process is examined, aiming to
robustly differentiate between earlier stages of the clinical spectrum of aging. Τhe target
classes of this study comprise Healthy controls, Subjective Cognitive Decline (SCD),
Early-MCI (E-MCI), Late-MCI (L-MCI).

To collect data, 84 persons, aged 50 to 85, were recorded at the Greek Association of
Alzheimer’s Disease and Related Disorders (GAADRD) center “Agia Eleni”, collecting
a total of 1621 recordings along with their personal information. The recording process
consisted of 5 different stages having the format of an informal interview with questions
and dual-task prompts, so as to steadily increase the required cognitive effort, aiming at
examining the performance differences across the stages.

Three different types of audio features were extracted: silence features, prosodic
features, and zero-crossings features. To quantify the changes in the participants’ speech
between stages, a new feature vector was formed by subtracting the individual feature
vectors between stages. The features per stage as well as the new features were evaluated
with three classifiers, namely Random Forest, Extra-Trees and Support Vector Machines.
Three sets of experiments were conducted according to the split of data in test and train
data. First two sets consist of experiments in a 4-classes-classification as described, with
random split of instances and split of instances per person accordingly, while the 3rd
set consists of binary classifiers for further examination of the models’ distinctive ability.
Different experiments were conducted, where models created by utilizing stage differences,
features per stage, or even used in an ensemble majority voting system. On the first set,
the best classification was achieved using models trained with stage differences features,
resulting to a mean accuracy of 80.99%, on the second set, the best classification was
achieved using models trained on features per stage reaching an accuracy of 60%, while
on the third set, the distinctive ability of the models was shown resulting to accuracies
of 94.1% for Healthy vs MCI, 91.4% for Healthy vs SCD and 71.5% for SCD vs MCI.

Kavelidis Frantzis Dimitrios

Electrical & Computer Engineering Department

Aristotle University of Thessaloniki, Greece

September 2022


## Experiments:

----------------------
Experiments set 1:
Random (80%-20%) split
----------------------
* Experiment 1: Classification per stage of recording
* Experiment 2: Classification using recordings/features from every stage
* Experiment 3: Classification per stage difference
* Experiment 4: Classification using Enseble Majority Voting System

----------------------
Experiments set 2:
LOSO-like split / Split per people (not using same people on test and train sets)
----------------------
* Experiment 5: Classification per stage of recording
* Experiment 6: Classification using recordings/features from every stage
* Experiment 7: Classification per stage difference
* Experiment 8: Classification using Enseble Majority Voting System

----------------------
Experiments set 3:
Binary classifiers / Split per people
----------------------
* Experiment 1 (hm,sm,hs): Classification per stage of recording (Binary: Healthy vs MCI, SCD vs MCI, Healthy vs SCD)
* Experiment 2 (hm,sm,hs): Classification using recordings/features from every stage (Binary: Healthy vs MCI, SCD vs MCI, Healthy vs SCD)
* Experiment 3 (hm,sm,hs): Classification using Enseble Majority Voting System (Binary: Healthy vs MCI, SCD vs MCI, Healthy vs SCD)

----------------------
Experiments set 4:
Additional Experiments
----------------------
* Diagnosis vs Human Identification ?! / Classification to 84 classes (number of participants), to check if the models learn to identify cognitive decline patterns or specific/personal features of voice / Showing issues of experiments set 1 regarding the indipendance of the instances
* Evaluation of questions

----------------------

## Explanations

To get Ensemble Majority Voting System's final decision (Experiments 4, 8):

* Step 1: feature_extraction.py to get all the csv files containing the features for each stage
* Step 2: completing_csv.py to fill in the demographic info
* Step 3: new_features_extraction.py to get all the csv files containing the features for each stage difference
* Step 4: completing_csv.py to fill in the demographic info
* Step 5: train_final_models.py to train all the models and scalers that are going to be used in the Enseble Majority Voting System
* Step 6: final_decision.py to get the results

For Experiment 3 in Binary classifiers (or maybe if you want to try using features per stage instead of features per stage difference), just ignore step 3 and 4.

For LOSO-like split, use leftout.py and leftout_original.py

Of course for confidentiality reasons and participants' privacy, the files of features as well as the recordings will remain private.

Interview protocol and concent form for participants are both available (in Greek).

## Status
As of the completion of the project, it will probably be modified in case of more research in the field. It may hold inaccuracies and could be implemented in more efficient ways regarding speed and recources management.

## Support - Contact
Reach out to me:
- dimitris.kave@gmail.com
