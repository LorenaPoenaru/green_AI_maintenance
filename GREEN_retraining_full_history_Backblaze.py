# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from datetime import date, datetime, timedelta
import pandas as pd

import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn import metrics, preprocessing
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from scipy import stats
import seaborn as sns
import os
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import time
import random

#Energy Measurement Tool
from codecarbon import EmissionsTracker


# -

def obtain_intervals(dataset):
    '''
    Generate interval terminals, so that samples in each interval have:
        interval_i = (timestamp >= terminal_i) and (timestamp < terminal_{i+1})

    Args:
        dataset (chr): Assuming only Backblaze (b) and Google (g) datasets exists
    '''
    if dataset == 'g':
        # time unit in Google: millisecond, tracing time: 29 days
        start_time = 604046279
        unit_period = 24 * 60 * 60 * 1000 * 1000  # unit period: one day
        end_time = start_time + 28*unit_period
    elif dataset == 'b':
        # time unit in Backblaze: month, tracing time: one year (12 months)
        start_time = 1
        unit_period = 1  # unit period: one month
        end_time = start_time + 12*unit_period
    
    # add one unit for the open-end of range function
    terminals = [i for i in range(start_time, end_time+unit_period, unit_period)]

    return terminals


def obtain_natural_chunks(features, labels, terminals):
    feature_list = []
    label_list = []
    for i in range(len(terminals) - 1):
        idx = np.logical_and(features[:, 0] >= terminals[i], features[:, 0] < terminals[i + 1])
        feature_list.append(features[idx][:, 1:])
        label_list.append(labels[idx])
    return feature_list, label_list



def downsampling(training_features, training_labels, random_seed, ratio=10):
    #return training_features, training_labels

    idx_true = np.where(training_labels == True)[0]
    idx_false = np.where(training_labels == False)[0]
    idx_false_resampled = resample(idx_false, n_samples=len(idx_true)*ratio, replace=False, random_state = random_seed)
    idx_resampled = np.concatenate([idx_false_resampled, idx_true])
    idx_resampled.sort()
    resampled_features = training_features[idx_resampled]
    resampled_labels = training_labels[idx_resampled]
    return resampled_features, resampled_labels


# Feature Importance Functions

def important_features_extraction(model, features_input):
    
    # extract features and their importances
    
    feature_importance_ranking = model.feature_importances_
    zipped_features = list(zip(feature_importance_ranking, features_input))
    sorted_features_zip = sorted(zipped_features, key = lambda x: x[0], reverse = True)
    
    # extract mean of importances
    
    importances = [i[0] for i in sorted_features_zip]
    mean_importances = np.mean(importances)
    
    # extract most important features and return
    
    most_important_features = [i[1] for i in sorted_features_zip if i[0]>= mean_importances]
    
    return most_important_features


def filtering_non_important_features(features_array, features_names, important_features_names):
    # transform array into dataframe and attach features
    df_features = pd.DataFrame(np.array(features_array), columns = features_names)
    
    # filter out columns with non-relevant features
    df_important_features = df_features[df_features.columns[~df_features.columns.isin(important_features)==0]]
    
    # transform dataframe with only into features back into array
    important_features_array = df_important_features.to_numpy()
    
    return important_features_array


def features_labels_preprocessing(DATASET_PATH, dataset):
    
    if(dataset=='b'):
        
        print('Data Reading and Preprocessing')
        
        # set data paths and columns names
        features_disk_failure = ['smart_1_raw', 'smart_4_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_12_raw', 'smart_187_raw', 'smart_193_raw', 'smart_194_raw', 'smart_197_raw', 'smart_199_raw', 
                         'smart_4_raw_diff', 'smart_5_raw_diff', 'smart_9_raw_diff', 'smart_12_raw_diff', 'smart_187_raw_diff', 'smart_193_raw_diff', 'smart_197_raw_diff', 'smart_199_raw_diff']
        columns = ['serial_number', 'date'] + features_disk_failure + ['label']
        
        # read dataset
        df = pd.read_csv(DATASET_PATH, header=None, dtype = 'str').iloc[1:,1:]
        df.columns = columns
        
        # ignore serial number
        df = df[df.columns[1:]]
        
        for feature in features_disk_failure:
            df[feature] = df[feature].astype(float)


        d = {'True': True, 'False': False}
        df['label'] = df['label'].map(d)

        df['label'].unique()

        # transform date to date time
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        # divide on weeks
        df['date'] = pd.Series(pd.DatetimeIndex(df['date']).day_of_year)
        
        print('Features and Labels Computing')
        
        # features and labels extraction and computation
        features = df[df.columns[:-1]].to_numpy()
        labels = df[df.columns[-1]].to_numpy()
        feature_list, label_list = obtain_natural_chunks(features, labels, obtain_intervals('b'))
        
    elif(dataset=='g'):
        
        print('Data Reading and Preprocessing')
        
        # set data paths and columns names
        features_job_failure = ['User ID', 'Job Name', 'Scheduling Class',
                   'Num Tasks', 'Priority', 'Diff Machine', 'CPU Requested', 'Mem Requested', 'Disk Requested',
                   'Avg CPU', 'Avg Mem', 'Avg Disk', 'Std CPU', 'Std Mem', 'Std Disk']
        columns_initial = ['Job ID', 'Status', 'Start Time', 'End Time'] + features_job_failure
        
        # read dataset
        df = pd.read_csv(DATASET_PATH, header=None)
        df.columns = columns_initial
        df = df.tail(-1)
        # ignore Job ID
        df = df.drop(['Job ID'], axis = 1)
        columns = features_job_failure

        include_end_time = False
        
        print('Features and Labels Preprocessing')
        
        # features and labels preprocessing
        features = df[(['Start Time']+ features_job_failure)].to_numpy()
        labels = (df['Status']==3).to_numpy()

        # FEATURES PREPROCESSING
        offset = (1 if include_end_time else 0)

        # ENCODE USER ID
        le = preprocessing.LabelEncoder()
        features[:, 1+offset] = le.fit_transform(features[:, 1+offset])

        # ENCODE JOB NAME
        le = preprocessing.LabelEncoder()
        features[:, 2+offset] = le.fit_transform(features[:, 2+offset])

        features = features.astype(float)
        
        print('Features and Labels Computing')
        
        # features and labels extraction and computation
        feature_list, label_list = obtain_natural_chunks(features, labels, obtain_intervals('g'))
        
    else:
        print('Incorrect Dataset')
    
    return feature_list, label_list


# +
def distribution_extraction(reference_data, testing_data, dataset_name, type_retraining_data, detection, random_seed, batch, total_distribution_tracker_values, tracker):
    tracker.start_task(set_name_tracker_for_task(dataset_name, type_retraining_data, detection, "Distribution_Extraction", random_seed, batch))   
    distribution_reference = sns.distplot(np.array(reference_data)).get_lines()[0].get_data()[1] ## Distrib extract Energy start
    plt.close()
    distribution_test = sns.distplot(np.array(testing_data)).get_lines()[0].get_data()[1]
    plt.close()
    distribution_emissions = tracker.stop_task()
    total_distribution_tracker_values['cpu'] += distribution_emissions.cpu_energy
    total_distribution_tracker_values['gpu'] += distribution_emissions.gpu_energy
    total_distribution_tracker_values['ram'] += distribution_emissions.ram_energy
    total_distribution_tracker_values['duration'] += distribution_emissions.duration
    return distribution_reference, distribution_test, total_distribution_tracker_values

def ks_stats(dataset_name, type_retraining_data, detection, random_seed, batch, distribution_reference, distribution_test, total_stats_tracker_values, tracker):
    
    tracker.start_task(set_name_tracker_for_task(dataset_name, type_retraining_data, detection, "KS_Stats", random_seed, batch))   
    stat_test = stats.kstest
    v, p = stat_test(distribution_reference, distribution_test) ### Stats Test Energy stop HERE!!!!!!!
    stats_emissions = tracker.stop_task()
    total_stats_tracker_values['cpu'] += stats_emissions.cpu_energy
    total_stats_tracker_values['gpu'] += stats_emissions.gpu_energy
    total_stats_tracker_values['ram'] += stats_emissions.ram_energy
    total_stats_tracker_values['duration'] += stats_emissions.duration
    return p, total_stats_tracker_values

def ks_drift_detection(dataset_name, type_retraining_data, detection, random_seed, batch, reference_data, testing_data, total_distribution_tracker_values, total_stats_tracker_values, tracker):
    
    # extract distributions from reference and testing data
    
    distribution_extraction_time_start = time.time()
    distribution_reference, distribution_test, total_distribution_tracker_values = distribution_extraction(reference_data, 
                                                                                                           testing_data, 
                                                                                                           dataset_name,
                                                                                                           type_retraining_data, 
                                                                                                           detection, 
                                                                                                           random_seed, 
                                                                                                           batch, 
                                                                                                           total_distribution_tracker_values,
                                                                                                           tracker)

    distribution_extraction_time_end = time.time() - distribution_extraction_time_start
    # apply KS statistical test
    
    ks_test_time_start = time.time() ### Stats Test Energy start HERE!!!!!!!
    p, total_stats_tracker_values = ks_stats(dataset_name, type_retraining_data, detection, random_seed, batch, distribution_reference, distribution_test, total_stats_tracker_values, tracker)
    ks_test_time_end = time.time() - ks_test_time_start
    # check if drift
    
    if(p<0.05):
        drift_alert = 1
    else:
        drift_alert = 0


    return drift_alert, distribution_extraction_time_end, ks_test_time_end, total_distribution_tracker_values, total_stats_tracker_values

def run_pca(dataset_name, type_retraining_data, detection, random_seed, batch, training_features, testing_features, total_pca_tracker_values, tracker):
    tracker.start_task(set_name_tracker_for_task(dataset_name, type_retraining_data, detection, "PCA", random_seed, batch))
    pca = PCA(n_components = 0.95, random_state = random_seed)
    pca.fit(training_features)
    df_train_features_sorted_pca = pca.transform(training_features)
    df_test_features_sorted_pca = pca.transform(testing_features)
    pca_emissions = tracker.stop_task()
    total_pca_tracker_values['cpu'] += pca_emissions.cpu_energy
    total_pca_tracker_values['gpu'] += pca_emissions.gpu_energy
    total_pca_tracker_values['ram'] += pca_emissions.ram_energy
    total_pca_tracker_values['duration'] += pca_emissions.duration
    return df_train_features_sorted_pca, df_test_features_sorted_pca, total_pca_tracker_values


def get_fi(dataset_name, type_retraining_data, detection, random_seed, batch, tracker, total_fi_tracker_values, update_model, training_features, testing_features, features_disk_failure):
    tracker.start_task(set_name_tracker_for_task(dataset_name, type_retraining_data, detection, "FI", random_seed, batch))
    important_features = important_features_extraction(update_model, features_disk_failure)
    # filter non-important features from train and test
    training_important_features_model = filtering_non_important_features(training_features, features_disk_failure, important_features)
    testing_important_features_model = filtering_non_important_features(testing_features, features_disk_failure, important_features)
    fi_emissions = tracker.stop_task()
    total_fi_tracker_values['cpu'] += fi_emissions.cpu_energy
    total_fi_tracker_values['gpu'] += fi_emissions.gpu_energy
    total_fi_tracker_values['ram'] += fi_emissions.ram_energy
    total_fi_tracker_values['duration'] += fi_emissions.duration
    return important_features, training_important_features_model, testing_important_features_model, total_fi_tracker_values

# Feature Importance Functions

def important_features_extraction(model, features_input):
    
    # extract features and their importances
    
    feature_importance_ranking = model.feature_importances_
    zipped_features = list(zip(feature_importance_ranking, features_input))
    sorted_features_zip = sorted(zipped_features, key = lambda x: x[0], reverse = True)
    
    # extract mean of importances
    
    importances = [i[0] for i in sorted_features_zip]
    mean_importances = np.mean(importances)
    
    # extract most important features and return
    
    most_important_features = [i[1] for i in sorted_features_zip if i[0]>= mean_importances]
    
    return most_important_features


def filtering_non_important_features(features_array, features_names, important_features_names):
    # transform array into dataframe and attach features
    df_features = pd.DataFrame(np.array(features_array), columns = features_names)
    
    # filter out columns with non-relevant features
    df_important_features = df_features[df_features.columns[~df_features.columns.isin(important_features_names)==0]]
    
    # transform dataframe with only into features back into array
    important_features_array = df_important_features.to_numpy()
    
    return important_features_array


def set_name_tracker_for_task(dataset_name, type_retraining_data, detection, task, random_seed, batch):
    return str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(task) + "_RandomSeed_" + str(random_seed) + "_Batch" + str(batch)


def hyperparameter_tuning_process(dataset_name, type_retraining_data, detection, random_seed, batch, param_dist_rf, N_ITER_SEARCH, training_features_downsampling, training_labels_downsampling, total_hyperparam_tracker_values, tracker):
    tracker.start_task(set_name_tracker_for_task(dataset_name, type_retraining_data, detection, "HyperparameterTuning", random_seed, batch))
    model = RandomForestClassifier(random_state = random_seed)
    random_search = RandomizedSearchCV(model,param_distributions = param_dist_rf,
                                            n_iter=N_ITER_SEARCH,
                                            scoring='roc_auc',
                                            cv=4, n_jobs=1, random_state = random_seed)
    random_search.fit(training_features_downsampling, training_labels_downsampling)
    update_model = random_search.best_estimator_
    hyperparameter_tuning_emissions = tracker.stop_task()
    total_hyperparam_tracker_values['cpu'] += hyperparameter_tuning_emissions.cpu_energy
    total_hyperparam_tracker_values['gpu'] += hyperparameter_tuning_emissions.gpu_energy
    total_hyperparam_tracker_values['ram'] += hyperparameter_tuning_emissions.ram_energy
    total_hyperparam_tracker_values['duration'] += hyperparameter_tuning_emissions.duration
    
    return update_model, total_hyperparam_tracker_values


def best_model_fit(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, training_features, training_labels, total_fit_tracker_values, tracker):
    tracker.start_task(set_name_tracker_for_task(dataset_name, type_retraining_data, detection, "ModelFit", random_seed, batch))
    update_model.fit(training_features, training_labels)
    model_fit_emissions = tracker.stop_task()
    total_fit_tracker_values['cpu'] += model_fit_emissions.cpu_energy
    total_fit_tracker_values['gpu'] += model_fit_emissions.gpu_energy
    total_fit_tracker_values['ram'] += model_fit_emissions.ram_energy
    total_fit_tracker_values['duration'] += model_fit_emissions.duration
    return update_model, total_fit_tracker_values


def format_data_for_the_seed(columns_names, values):    
    df_results_periodic_fh = pd.DataFrame(columns=columns_names)
    df_results_periodic_fh.loc[0] = values
    return df_results_periodic_fh


def store_into_file(filename, df_results_periodic_fh):
    df_results_periodic_fh.to_csv(filename, index=False,  sep=";") #, mode='a', header=not os.path.exists(filename))


def initiate_training_features_and_labels_from_lists(feature_list, label_list, num_chunks):
    # obtain training features and labels
    training_features = np.vstack(feature_list[0: num_chunks//2])
    training_labels = np.hstack(label_list[0//2: num_chunks//2])
    return training_features, training_labels

def get_testing_features_and_labels_from_lists(feature_list, label_list, batch):
    testing_features = feature_list[batch]
    testing_labels = label_list[batch] 
    return testing_features, testing_labels

def scaling_data(training_features, testing_features):
    # scaler for training data
    update_scaler = StandardScaler()
    training_features = update_scaler.fit_transform(training_features)
    # scaling testing features
    testing_features = update_scaler.transform(testing_features)
    return training_features, testing_features

def get_predictions(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, testing_features, total_testing_tracker_values, tracker):
    tracker.start_task(set_name_tracker_for_task(dataset_name, type_retraining_data, detection, "Testing", random_seed, batch))
    predictions_test_updated = update_model.predict(testing_features)
    testing_emissions = tracker.stop_task()
    total_testing_tracker_values['cpu'] += testing_emissions.cpu_energy
    total_testing_tracker_values['gpu'] += testing_emissions.gpu_energy
    total_testing_tracker_values['ram'] += testing_emissions.ram_energy
    total_testing_tracker_values['duration'] += testing_emissions.duration
    return predictions_test_updated, total_testing_tracker_values

def initiate_tracker_var():
    return {'cpu': 0, 'gpu': 0, 'ram':0, 'duration':0}
def initiate_tracker_variables():    
    total_hyperparam_tracker_values = initiate_tracker_var()
    total_fit_tracker_values = initiate_tracker_var()
    total_testing_tracker_values = initiate_tracker_var()
    return total_hyperparam_tracker_values, total_fit_tracker_values, total_testing_tracker_values


def process_for_a_batch(batch, feature_list, label_list, dataset_name, type_retraining_data, detection, random_seed, param_dist_rf, N_ITER_SEARCH, training_features, 
                                                                                     training_labels, length_training_fh, total_train_fh, 
                                                                                     total_test_fh, total_hyperparam_fh, partial_roc_auc_fh, 
                                                                                     predictions_test_fh, total_hyperparam_tracker_values, 
                                                                                     total_fit_tracker_values, total_testing_tracker_values, tracker):
    # obtain features and labels
    testing_features, testing_labels = get_testing_features_and_labels_from_lists(feature_list, label_list, batch)
    # scaling data
    training_features, testing_features = scaling_data(training_features, testing_features)
    # Downscaling for data training
    training_features_downsampling, training_labels_downsampling = downsampling(training_features, training_labels, random_seed)
        
    length_training_fh = length_training_fh + len(training_features_downsampling)
        
    # training model
    begin_train_fh = time.time()

    # Hyperparameter tunning energy collection start
    begin_hyperparam_tunning_update = time.time()
    update_model, total_hyperparam_tracker_values = hyperparameter_tuning_process(dataset_name, type_retraining_data, detection, random_seed, batch, param_dist_rf, N_ITER_SEARCH, training_features_downsampling, training_labels_downsampling,total_hyperparam_tracker_values, tracker)
    end_hyperparam_tunning_update = time.time() - begin_hyperparam_tunning_update

    #print('Training')
    update_model, total_fit_tracker_values = best_model_fit(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, training_features, training_labels, total_fit_tracker_values, tracker)
    end_train_fh = time.time() - begin_train_fh 
    total_hyperparam_fh = total_hyperparam_fh + end_hyperparam_tunning_update
    total_train_fh = total_train_fh + end_train_fh
        
        
    # evaluate model on testing data
    begin_test_fh = time.time()
    predictions_test_updated, total_testing_tracker_values = get_predictions(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, testing_features, total_testing_tracker_values, tracker)
    end_test_fh = time.time() - begin_test_fh     
    total_test_fh = total_test_fh + end_test_fh

    partial_roc_auc_fh.append(roc_auc_score(testing_labels, predictions_test_updated))
    predictions_test_fh = np.concatenate([predictions_test_fh, predictions_test_updated])

    training_features = np.vstack(feature_list[0: batch+1])
    training_labels = np.hstack(label_list[0: batch+1])
    
    return partial_roc_auc_fh, predictions_test_fh, total_train_fh, total_hyperparam_fh, total_test_fh, length_training_fh, total_hyperparam_tracker_values, total_fit_tracker_values, total_testing_tracker_values 


def pipeline_periodic_model_for_a_seed(dataset_name, type_retraining_data, detection, random_seed, feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels):
    
    experiment_name = str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(random_seed) + ".csv"

    predictions_test_fh = []
    partial_roc_auc_fh = []

    total_train_fh = 0 
    total_hyperparam_fh = 0
    total_test_fh = 0
    length_training_fh = 0
    
    tracker = EmissionsTracker(project_name="AI_maintenance_" + str(experiment_name))
    total_hyperparam_tracker_values, total_fit_tracker_values, total_testing_tracker_values = initiate_tracker_variables()
    
    begin_total_fh = time.time()
    
    training_features, training_labels = initiate_training_features_and_labels_from_lists(feature_list, label_list, num_chunks)

    for batch in tqdm(range(num_chunks//2, num_chunks)):
        
        partial_roc_auc_fh, predictions_test_fh, total_train_fh, total_hyperparam_fh, total_test_fh, length_training_fh, total_hyperparam_tracker_values, total_fit_tracker_values, total_testing_tracker_values  = process_for_a_batch(batch, feature_list, label_list, dataset_name, 
                                                                                     type_retraining_data, detection, random_seed, 
                                                                                     param_dist_rf, N_ITER_SEARCH, training_features, 
                                                                                     training_labels, length_training_fh, total_train_fh, 
                                                                                     total_test_fh, total_hyperparam_fh, partial_roc_auc_fh, 
                                                                                     predictions_test_fh, total_hyperparam_tracker_values, 
                                                                                     total_fit_tracker_values, total_testing_tracker_values, tracker)

        
        #print('Length of Training', length_training_fh)

    end_total_fh = time.time() - begin_total_fh
    
    columns_names = ['Random_Seed', 'Model', 'Drifts', 'ROC_AUC_Batch', 'ROC_AUC_BATCH_MEAN', 'ROC_AUC_Total', 'Predictions', 'True_Testing_Labels', 'Train_Time', 'Hyperparam_Tunning_Time', 'Test_Time', 'Drifts_Detected', 'Label_Costs', 'CPU_Energy_Hyperparameter', 'GPU_Energy_Hyperparameter', 'RAM_Energy_Hyperparameter', 'Duration_Tracker_Hyperparameter', 'CPU_Energy_Fitting', 'GPU_Energy_Fitting', 'RAM_Energy_Fitting', 'Duration_Tracker_Fitting', 'CPU_Energy_Testing', 'GPU_Energy_Testing', 'RAM_Energy_Testing', 'Duration_Tracker_Testing']
    values = [random_seed, 'periodic-sw', str(int(num_chunks//2)) + '/' + str(int(num_chunks//2)), partial_roc_auc_fh, np.mean(partial_roc_auc_fh), roc_auc_score(true_testing_labels, predictions_test_fh), predictions_test_fh, true_testing_labels, total_train_fh, total_hyperparam_fh, total_test_fh, np.ones(int(num_chunks//2), dtype=int), len(true_testing_labels), 
              total_hyperparam_tracker_values['cpu'], total_hyperparam_tracker_values['gpu'], total_hyperparam_tracker_values['ram'], total_hyperparam_tracker_values['duration'], total_fit_tracker_values['cpu'], total_fit_tracker_values['gpu'], total_fit_tracker_values['ram'], total_fit_tracker_values['duration'], total_testing_tracker_values['cpu'], total_testing_tracker_values['gpu'], total_testing_tracker_values['ram'], total_testing_tracker_values['duration']]
    df_results_for_seed = format_data_for_the_seed(columns_names, values)
    store_into_file('./results/Output_' + str(experiment_name), df_results_for_seed)
    _ = tracker.stop()


# # # Build Drift Detection based Model Update
# # ### KS on all features

def pipeline_ks_all(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list):
    experiment_name = str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(random_seed) + ".csv"


    necessary_label_annotation_effort = 0
    total_time_training = 0
    no_necessary_retrainings = 0
    lengths_training_ks_all = []
    partial_roc_auc_ks_all_model = []
    predictions_test_ks_all_model = []
    

    total_train_fh_all = 0
    total_hyperparam_fh_ks_all = 0
    total_test_time_ks_all = 0
    
    total_drift_detection_time = 0
    total_distribution_extraction_time = 0
    total_stat_test_time = 0
    
    
    detected_drifts = []
    
    ### Tracker tasks variables 
    tracker = EmissionsTracker(project_name="AI_maintenance_" + str(experiment_name))
    total_hyperparam_tracker_values, total_fit_tracker_values, total_testing_tracker_values = initiate_tracker_variables()
    total_distribution_tracker_values = initiate_tracker_var()
    total_stats_tracker_values = initiate_tracker_var()
    

    training_features, training_labels = initiate_training_features_and_labels_from_lists(feature_list, label_list, num_chunks)
    current_training_batches_list = initial_training_batches_list.copy()
    print('Initial Training Batches', current_training_batches_list)
    #need_to_retrain = 0
    for batch in tqdm(range(num_chunks//2, num_chunks)):
        
        # init drift alert
        drift_alert = 0
        
        # obtain features and labels
        testing_features, testing_labels = get_testing_features_and_labels_from_lists(feature_list, label_list, batch)
        # scaling data
        training_features, testing_features = scaling_data(training_features, testing_features)
        # Downscaling for data training
        training_features, training_labels = downsampling(training_features, training_labels, random_seed)

         # training model
        begin_train_fh_ks_all = time.time()


        if(batch==num_chunks//2 or need_to_retrain == 1):
           
            begin_train_fh_ks_all = time.time()
            begin_hyperparam_tunning_update = time.time()
            update_model, total_hyperparam_tracker_values = hyperparameter_tuning_process(dataset_name, type_retraining_data, 
                                                                                          detection, random_seed, batch, param_dist_rf, 
                                                                                          N_ITER_SEARCH, training_features, 
                                                                                          training_labels,
                                                                                          total_hyperparam_tracker_values, tracker)
            
            end_hyperparam_tunning_update = time.time() - begin_hyperparam_tunning_update
            total_hyperparam_fh_ks_all = total_hyperparam_fh_ks_all + end_hyperparam_tunning_update

            update_model, total_fit_tracker_values = best_model_fit(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, training_features, training_labels, total_fit_tracker_values, tracker)
            end_train_fh_ks_all = time.time() - begin_train_fh_ks_all
            total_train_fh_all = total_train_fh_all + end_train_fh_ks_all
        
        
        # evaluate model on testing data
        begin_test_time_ks_all = time.time()
        predictions_test_updated, total_testing_tracker_values = get_predictions(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, testing_features, total_testing_tracker_values, tracker)
        end_test_time_ks_all = time.time() - begin_test_time_ks_all
        total_test_time_ks_all = total_test_time_ks_all + end_test_time_ks_all

        partial_roc_auc_ks_all_model.append(roc_auc_score(testing_labels, predictions_test_updated)) 
        predictions_test_ks_all_model = np.concatenate([predictions_test_ks_all_model, predictions_test_updated])
        
        
        # Drift Detection
        
        need_to_retrain = 0        
        
        drift_time_start = time.time()
        drift_alert, distribution_extraction_time, ks_test_time, total_distribution_tracker_values, total_stats_tracker_values = ks_drift_detection(dataset_name, type_retraining_data, detection, random_seed, batch, training_features, testing_features, total_distribution_tracker_values, total_stats_tracker_values, tracker)
        drift_time_end = time.time() - drift_time_start
        
        
        total_distribution_extraction_time = total_distribution_extraction_time + distribution_extraction_time
        total_stat_test_time = total_stat_test_time + ks_test_time
        total_drift_detection_time = total_drift_detection_time + drift_time_end
        
        detected_drifts.append(drift_alert)
                
        if(drift_alert==1):
        
            need_to_retrain = 1
            drift_alert = 0

            no_necessary_retrainings = no_necessary_retrainings + 1
            necessary_label_annotation_effort = necessary_label_annotation_effort + len(testing_labels)

            # add new data to the training for full history approach
            current_training_batches_list.append(batch)
                    
            training_features_list_updated = [feature_list[i] for i in current_training_batches_list]
            training_labels_list_updated = [label_list[i] for i in current_training_batches_list]
        
            training_features = np.vstack(training_features_list_updated)
            training_labels = np.hstack(training_labels_list_updated)

        
        #print('Current Training Batches',current_training_batches_list)
    columns_names = ['Random_Seed', 'Model', 'Drifts_Overall',  'ROC_AUC_Batch', 'ROC_AUC_BATCH_MEAN', 'ROC_AUC_Total', 
                     'Predictions', 'True_Testing_Labels', 'Train_Time', 'Hyperparam_Tunning_Time', 'Test_Time', 
                     'Drifts_Detected', 'Drift_Detection_Total_Time', 'Distribution_Extraction_Time', 'Statistical_Test_Time', 'Label_Costs', 
                     'CPU_Energy_Hyperparameter', 'GPU_Energy_Hyperparameter', 'RAM_Energy_Hyperparameter', 'Duration_Tracker_Hyperparameter', 
                     'CPU_Energy_Fitting', 'GPU_Energy_Fitting', 'RAM_Energy_Fitting', 'Duration_Tracker_Fitting', 
                     'CPU_Energy_Testing', 'GPU_Energy_Testing', 'RAM_Energy_Testing', 'Duration_Tracker_Testing',
                     'CPU_Energy_Distribution_Extraction', 'GPU_Energy_Distribution_Extraction', 'RAM_Energy_Distribution_Extraction', 'Duration_Tracker_Distribution_Extraction',
                     'CPU_Energy_Stats_Test', 'GPU_Energy_Stats_Test', 'RAM_Energy_Stats_Test', 'Duration_Tracker_Stats_Test'
                     ]
    values = [random_seed, detection, str(no_necessary_retrainings)+'/'+str(len(detected_drifts)), partial_roc_auc_ks_all_model, 
              np.mean(partial_roc_auc_ks_all_model), roc_auc_score(true_testing_labels, predictions_test_ks_all_model), 
              predictions_test_ks_all_model, true_testing_labels, total_train_fh_all, total_hyperparam_fh_ks_all, 
              total_test_time_ks_all, detected_drifts, total_drift_detection_time, total_distribution_extraction_time, 
              total_stat_test_time, necessary_label_annotation_effort,
              total_hyperparam_tracker_values['cpu'], total_hyperparam_tracker_values['gpu'], total_hyperparam_tracker_values['ram'], total_hyperparam_tracker_values['duration'], 
              total_fit_tracker_values['cpu'], total_fit_tracker_values['gpu'], total_fit_tracker_values['ram'], total_fit_tracker_values['duration'], 
              total_testing_tracker_values['cpu'], total_testing_tracker_values['gpu'], total_testing_tracker_values['ram'], total_testing_tracker_values['duration'],
              total_distribution_tracker_values['cpu'], total_distribution_tracker_values['gpu'], total_distribution_tracker_values['ram'], total_distribution_tracker_values['duration'],
              total_stats_tracker_values['cpu'], total_stats_tracker_values['gpu'], total_stats_tracker_values['ram'], total_stats_tracker_values['duration']]
    
    df_results_for_seed = format_data_for_the_seed(columns_names, values)
    store_into_file('./results/Output_' + str(experiment_name), df_results_for_seed)
    


def pipeline_ks_pca(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list):
    experiment_name = str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(random_seed) + ".csv"

#     print('Random Seed:', random_seed)
    necessary_label_annotation_effort = 0
    no_necessary_retrainings = 0
    
    partial_roc_auc_ks_pca_model = []
    predictions_test_ks_pca_model = []

    total_train_fh_pca = 0
    total_hyperparam_fh_ks_pca = 0
    total_test_time_ks_pca = 0
    
    total_drift_detection_time = 0
    total_distribution_extraction_time = 0
    total_stat_test_time = 0
    total_pca_time = 0
    
    
    detected_drifts = []


    ### Tracker tasks variables 
    tracker = EmissionsTracker(project_name="AI_maintenance_" + str(experiment_name))
    total_hyperparam_tracker_values, total_fit_tracker_values, total_testing_tracker_values = initiate_tracker_variables()
    total_distribution_tracker_values = initiate_tracker_var()
    total_stats_tracker_values = initiate_tracker_var()
    total_pca_tracker_values = initiate_tracker_var()
    

    training_features, training_labels = initiate_training_features_and_labels_from_lists(feature_list, label_list, num_chunks)
    current_training_batches_list = initial_training_batches_list.copy()
    #need_to_retrain = 0
    for batch in tqdm(range(num_chunks//2, num_chunks)):
        
        # init drift alert
        drift_alert = 0
        
        # obtain features and labels
        testing_features, testing_labels = get_testing_features_and_labels_from_lists(feature_list, label_list, batch)
        # scaling data
        training_features, testing_features = scaling_data(training_features, testing_features)
        # Downscaling for data training
        training_features, training_labels = downsampling(training_features, training_labels, random_seed)
    
        begin_train_fh_ks_pca = time.time()


        if(batch==num_chunks//2 or need_to_retrain == 1):
            begin_train_fh_ks_pca = time.time()
            begin_hyperparam_tunning_update = time.time()
            update_model, total_hyperparam_tracker_values = hyperparameter_tuning_process(dataset_name, type_retraining_data, 
                                                                                          detection, random_seed, batch, param_dist_rf, 
                                                                                          N_ITER_SEARCH, training_features, 
                                                                                          training_labels,
                                                                                          total_hyperparam_tracker_values, tracker)
            end_hyperparam_tunning_update = time.time() - begin_hyperparam_tunning_update
            
            total_hyperparam_fh_ks_pca = total_hyperparam_fh_ks_pca + end_hyperparam_tunning_update
            
            update_model, total_fit_tracker_values = best_model_fit(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, training_features, training_labels, total_fit_tracker_values, tracker)

            end_train_fh_ks_pca = time.time() - begin_train_fh_ks_pca
            total_train_fh_pca = total_train_fh_pca + end_train_fh_ks_pca
        
        
        # evaluate model on testing data & measure testing time
        begin_test_time_ks_pca = time.time()
        predictions_test_updated, total_testing_tracker_values = get_predictions(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, testing_features, total_testing_tracker_values, tracker)

        end_test_time_ks_pca = time.time() - begin_test_time_ks_pca
        total_test_time_ks_pca = total_test_time_ks_pca + end_test_time_ks_pca

        
        # ROC AUC
        partial_roc_auc_ks_pca_model.append(roc_auc_score(testing_labels, predictions_test_updated))
        predictions_test_ks_pca_model = np.concatenate([predictions_test_ks_pca_model, predictions_test_updated])
        
        # Drift Detection
        
        need_to_retrain = 0
        drift_time_start = time.time()
        
        # Extract PCA Features
        
        pca_computing_time_start = time.time()
        df_train_features_sorted_pca, df_test_features_sorted_pca, total_pca_tracker_values =run_pca(dataset_name, type_retraining_data, detection, random_seed, batch, training_features, testing_features, total_pca_tracker_values, tracker)
        pca_computing_time_end = time.time() - pca_computing_time_start
        
#         # Detect Drift
        drift_alert, distribution_extraction_time, ks_test_time, total_distribution_tracker_values, total_stats_tracker_values = ks_drift_detection(dataset_name, type_retraining_data, detection, random_seed, batch, df_train_features_sorted_pca, df_test_features_sorted_pca, total_distribution_tracker_values, total_stats_tracker_values, tracker)
        drift_time_end = time.time() - drift_time_start
        
        
        
        
        
        total_distribution_extraction_time = total_distribution_extraction_time + distribution_extraction_time
        total_stat_test_time = total_stat_test_time + ks_test_time
        total_pca_time = total_pca_time + pca_computing_time_end
        total_drift_detection_time = total_drift_detection_time + drift_time_end
        
        detected_drifts.append(drift_alert)
        
        if(drift_alert==1):
        
            need_to_retrain = 1
            drift_alert = 0


            no_necessary_retrainings = no_necessary_retrainings + 1
            necessary_label_annotation_effort = necessary_label_annotation_effort + len(testing_labels)

            
            # add new data to the training for full history approach
            current_training_batches_list.append(batch)
                    
            
            training_features_list_updated = [feature_list[i] for i in current_training_batches_list]
            training_labels_list_updated = [label_list[i] for i in current_training_batches_list]
        
            training_features = np.vstack(training_features_list_updated)
            training_labels = np.hstack(training_labels_list_updated)

        
    
    columns_names=['Random_Seed', 'Model', 'Drifts_Overall',  'ROC_AUC_Batch', 
    'ROC_AUC_BATCH_MEAN', 'ROC_AUC_Total', 'Predictions', 
    'True_Testing_Labels', 'Train_Time', 'Hyperparam_Tunning_Time', 
    'Test_Time', 'Drifts_Detected', 'Drift_Detection_Total_Time', 
    'PCA_Computing_time', 'Distribution_Extraction_Time', 'Statistical_Test_Time', 'Label_Costs',
    'CPU_Energy_Hyperparameter', 'GPU_Energy_Hyperparameter', 'RAM_Energy_Hyperparameter', 'Duration_Tracker_Hyperparameter', 
    'CPU_Energy_Fitting', 'GPU_Energy_Fitting', 'RAM_Energy_Fitting', 'Duration_Tracker_Fitting', 
    'CPU_Energy_Testing', 'GPU_Energy_Testing', 'RAM_Energy_Testing', 'Duration_Tracker_Testing',
    'CPU_Energy_Distribution_Extraction', 'GPU_Energy_Distribution_Extraction', 'RAM_Energy_Distribution_Extraction', 'Duration_Tracker_Distribution_Extraction',
    'CPU_Energy_Stats_Test', 'GPU_Energy_Stats_Test', 'RAM_Energy_Stats_Test', 'Duration_Tracker_Stats_Test',
    'CPU_Energy_PCA', 'GPU_Energy_PCA', 'RAM_Energy_PCA', 'Duration_Tracker_PCA']
    values=[random_seed, 'KS_PCA', str(no_necessary_retrainings)+'/'+str(len(detected_drifts)), 
    partial_roc_auc_ks_pca_model, np.mean(partial_roc_auc_ks_pca_model), 
    roc_auc_score(true_testing_labels, predictions_test_ks_pca_model), 
    predictions_test_ks_pca_model, true_testing_labels, total_train_fh_pca, 
    total_hyperparam_fh_ks_pca, total_test_time_ks_pca, detected_drifts, 
    total_drift_detection_time, total_pca_time, total_distribution_extraction_time, 
    total_stat_test_time, necessary_label_annotation_effort,
    total_hyperparam_tracker_values['cpu'], total_hyperparam_tracker_values['gpu'], total_hyperparam_tracker_values['ram'], total_hyperparam_tracker_values['duration'],    
    total_fit_tracker_values['cpu'], total_fit_tracker_values['gpu'], total_fit_tracker_values['ram'], total_fit_tracker_values['duration'], 
    total_testing_tracker_values['cpu'], total_testing_tracker_values['gpu'], total_testing_tracker_values['ram'], total_testing_tracker_values['duration'],
    total_distribution_tracker_values['cpu'], total_distribution_tracker_values['gpu'], total_distribution_tracker_values['ram'], total_distribution_tracker_values['duration'],
    total_stats_tracker_values['cpu'], total_stats_tracker_values['gpu'], total_stats_tracker_values['ram'], total_stats_tracker_values['duration'],
    total_pca_tracker_values['cpu'], total_pca_tracker_values['gpu'], total_pca_tracker_values['ram'], total_pca_tracker_values['duration']]

    df_results_for_seed = format_data_for_the_seed(columns_names, values)
    store_into_file('./results/Output_' + str(experiment_name), df_results_for_seed)
    


# # # KS on Most Important Features



# # +
# initial_training_batches_list = list(range(0, num_chunks//2))


# for random_seed in random_seeds:

def pipeline_ks_fi(features_disk_failure, dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list):
    experiment_name = str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(random_seed) + ".csv"

    no_necessary_retrainings = 0
    necessary_label_annotation_effort = 0
    
    partial_roc_auc_ks_fi_model = []    
    predictions_test_ks_fi_model = []
    
    total_train_fh_fi = 0
    total_hyperparam_fh_ks_fi = 0
    total_test_time_ks_fi = 0
    
    total_feature_importance_extraction_time = 0
    total_distribution_extraction_time = 0
    total_stat_test_time = 0
    
    total_drift_detection_time = 0

    detected_drifts = []

    ### Tracker tasks variables 
    tracker = EmissionsTracker(project_name="AI_maintenance_" + str(experiment_name))
    total_hyperparam_tracker_values, total_fit_tracker_values, total_testing_tracker_values = initiate_tracker_variables()
    total_distribution_tracker_values = initiate_tracker_var()
    total_stats_tracker_values = initiate_tracker_var()
    total_fi_tracker_values = initiate_tracker_var()
    

    training_features, training_labels = initiate_training_features_and_labels_from_lists(feature_list, label_list, num_chunks)
    current_training_batches_list = initial_training_batches_list.copy()
    #need_to_retrain = 0
    for batch in tqdm(range(num_chunks//2, num_chunks)):
       # init drift alert
        drift_alert = 0
        
        # obtain features and labels
        testing_features, testing_labels = get_testing_features_and_labels_from_lists(feature_list, label_list, batch)
        # scaling data
        training_features, testing_features = scaling_data(training_features, testing_features)
        # Downscaling for data training
        training_features, training_labels = downsampling(training_features, training_labels, random_seed)

        # training model
        begin_train_fh_ks_fi = time.time()


        if(batch==num_chunks//2 or need_to_retrain == 1):
 
            begin_train_fh_ks_fi = time.time()
        
            begin_hyperparam_tunning_update = time.time()
            update_model, total_hyperparam_tracker_values = hyperparameter_tuning_process(dataset_name, type_retraining_data, 
                                                                                          detection, random_seed, batch, param_dist_rf, 
                                                                                          N_ITER_SEARCH, training_features, 
                                                                                          training_labels,
                                                                                          total_hyperparam_tracker_values, tracker)
            end_hyperparam_tunning_update = time.time() - begin_hyperparam_tunning_update
            total_hyperparam_fh_ks_fi = total_hyperparam_fh_ks_fi + end_hyperparam_tunning_update
            
            
            
            
            update_model, total_fit_tracker_values = best_model_fit(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, training_features, training_labels, total_fit_tracker_values, tracker)
            
            end_train_fh_ks_fi = time.time() - begin_train_fh_ks_fi
            total_train_fh_fi = total_train_fh_fi + end_train_fh_ks_fi
        
        
#         # evaluate model on testing data
        
        begin_test_time_ks_fi = time.time()
        predictions_test_updated, total_testing_tracker_values = get_predictions(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, testing_features, total_testing_tracker_values, tracker)
        end_test_time_ks_fi = time.time() - begin_test_time_ks_fi
        total_test_time_ks_fi = total_test_time_ks_fi + end_test_time_ks_fi
        
        partial_roc_auc_ks_fi_model.append(roc_auc_score(testing_labels, predictions_test_updated))
        predictions_test_ks_fi_model = np.concatenate([predictions_test_ks_fi_model, predictions_test_updated])
        

#         # Drift Detection
        
        need_to_retrain = 0
        

        
        drift_time_start = time.time()
        
        # Extract Most Important Features
        feature_importance_extraction_start = time.time()  

        important_features, training_important_features_model, testing_important_features_model, total_fi_tracker_values = get_fi(dataset_name, type_retraining_data, detection, random_seed, batch, tracker, total_fi_tracker_values, update_model, training_features, testing_features, features_disk_failure)

        feature_importance_extraction_end = time.time() - feature_importance_extraction_start
        
        
#         # Detect Drift

        drift_alert, distribution_extraction_time, ks_test_time, total_distribution_tracker_values, total_stats_tracker_values = ks_drift_detection(dataset_name, type_retraining_data, detection, random_seed, batch, training_important_features_model, testing_important_features_model, total_distribution_tracker_values, total_stats_tracker_values, tracker)
        drift_time_end = time.time() - drift_time_start
        
        
        total_distribution_extraction_time = total_distribution_extraction_time + distribution_extraction_time
        total_stat_test_time = total_stat_test_time + ks_test_time
        total_feature_importance_extraction_time = total_feature_importance_extraction_time + feature_importance_extraction_end
        total_drift_detection_time = total_drift_detection_time + drift_time_end
        
        detected_drifts.append(drift_alert)
        
        if(drift_alert==1):
            need_to_retrain = 1
            drift_alert = 0

            no_necessary_retrainings = no_necessary_retrainings + 1
            necessary_label_annotation_effort = necessary_label_annotation_effort + len(testing_labels)
            
            # add new data to the training for full history approach
            current_training_batches_list.append(batch)
            
            training_features_list_updated = [feature_list[i] for i in current_training_batches_list]
            training_labels_list_updated = [label_list[i] for i in current_training_batches_list]
        
            training_features = np.vstack(training_features_list_updated)
            training_labels = np.hstack(training_labels_list_updated)


    
    
    columns_names =['Random_Seed', 'Model', 'Drifts_Overall',  'ROC_AUC_Batch', 'ROC_AUC_BATCH_MEAN', 
    'ROC_AUC_Total', 'Predictions', 'True_Testing_Labels', 'Train_Time', 'Hyperparam_Tunning_Time', 
    'Test_Time', 'Drifts_Detected', 'Drift_Detection_Total_Time', 'FI_Extraction_Time', 
    'Distribution_Extraction_Time', 'Statistical_Test_Time', 'Label_Costs',
        'CPU_Energy_Hyperparameter', 'GPU_Energy_Hyperparameter', 'RAM_Energy_Hyperparameter', 'Duration_Tracker_Hyperparameter', 
    'CPU_Energy_Fitting', 'GPU_Energy_Fitting', 'RAM_Energy_Fitting', 'Duration_Tracker_Fitting', 
    'CPU_Energy_Testing', 'GPU_Energy_Testing', 'RAM_Energy_Testing', 'Duration_Tracker_Testing',
    'CPU_Energy_Distribution_Extraction', 'GPU_Energy_Distribution_Extraction', 'RAM_Energy_Distribution_Extraction', 'Duration_Tracker_Distribution_Extraction',
    'CPU_Energy_Stats_Test', 'GPU_Energy_Stats_Test', 'RAM_Energy_Stats_Test', 'Duration_Tracker_Stats_Test',
    'CPU_Energy_FI', 'GPU_Energy_FI', 'RAM_Energy_FI', 'Duration_Tracker_FI']
    values= [random_seed, 'KS_FI', str(no_necessary_retrainings)+'/'+str(len(detected_drifts)), 
    partial_roc_auc_ks_fi_model, np.mean(partial_roc_auc_ks_fi_model), roc_auc_score(true_testing_labels, 
    predictions_test_ks_fi_model), predictions_test_ks_fi_model, true_testing_labels, total_train_fh_fi, 
    total_hyperparam_fh_ks_fi, total_test_time_ks_fi, detected_drifts, total_drift_detection_time, total_feature_importance_extraction_time, 
    total_distribution_extraction_time, total_stat_test_time, necessary_label_annotation_effort,
    total_hyperparam_tracker_values['cpu'], total_hyperparam_tracker_values['gpu'], total_hyperparam_tracker_values['ram'], total_hyperparam_tracker_values['duration'],    
    total_fit_tracker_values['cpu'], total_fit_tracker_values['gpu'], total_fit_tracker_values['ram'], total_fit_tracker_values['duration'], 
    total_testing_tracker_values['cpu'], total_testing_tracker_values['gpu'], total_testing_tracker_values['ram'], total_testing_tracker_values['duration'],
    total_distribution_tracker_values['cpu'], total_distribution_tracker_values['gpu'], total_distribution_tracker_values['ram'], total_distribution_tracker_values['duration'],
    total_stats_tracker_values['cpu'], total_stats_tracker_values['gpu'], total_stats_tracker_values['ram'], total_stats_tracker_values['duration'],
    total_fi_tracker_values['cpu'], total_fi_tracker_values['gpu'], total_fi_tracker_values['ram'], total_fi_tracker_values['duration']]
    
    df_results_for_seed = format_data_for_the_seed(columns_names, values)
    store_into_file('./results/Output_' + str(experiment_name), df_results_for_seed)
    


def main(): 
    dataset_name = "Backblaze"
    type_retraining_data = "FullHistory"
    DATASET_PATH_DISK = "./disk_2015_complete.csv"
    print(DATASET_PATH_DISK)
    feature_list, label_list = features_labels_preprocessing(DATASET_PATH_DISK, 'b')
    num_chunks = len(feature_list)
    true_testing_labels = np.hstack(label_list[num_chunks//2:])
    initial_training_batches_list = list(range(0, num_chunks//2))
    features_disk_failure = ['smart_1_raw', 'smart_4_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_12_raw', 'smart_187_raw', 'smart_193_raw', 'smart_194_raw', 'smart_197_raw', 'smart_199_raw', 
                         'smart_4_raw_diff', 'smart_5_raw_diff', 'smart_9_raw_diff', 'smart_12_raw_diff', 'smart_187_raw_diff', 'smart_193_raw_diff', 'smart_197_raw_diff', 'smart_199_raw_diff']


    # Hyperparameter tuning parameter
    param_dist_rf = {
                'n_estimators': stats.randint(1e1, 1e2),
                'criterion': ['gini', 'entropy'],
                'max_features': ['sqrt', 'log2'],
                'max_depth': [int(x) for x in np.linspace(10, 110, num=6)] + [None],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 4, 8],
                'class_weight':['balanced', None],
                'bootstrap': [True, False]
            }

    N_WORKERS = 1
    TOTAL_NUMBER_SEEDS = 2
    random_seeds = list(np.arange(TOTAL_NUMBER_SEEDS))
    N_ITER_SEARCH = 100
    detections = ["KS-FI"] #"KS-PCA", "PeriodicModel",  "KS-ALL"]

    counter = {}
    for detection in detections:
        counter[detection] = TOTAL_NUMBER_SEEDS


    executions = detections * TOTAL_NUMBER_SEEDS
    random.shuffle(executions)
    print(executions)

    for detection in executions:
        print(detection)
        random_seed = random_seeds[counter[detection]-1]
        if detection == "PeriodicModel":
            pipeline_periodic_model_for_a_seed(dataset_name, type_retraining_data, detection, random_seed, feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels)
        if detection == "KS-ALL":
            pipeline_ks_all(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
        if detection == "KS-PCA":
            pipeline_ks_pca(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
        if detection == "KS-FI":
            pipeline_ks_fi(features_disk_failure, dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
        counter[detection] -= 1
    print("End of Experimentation")


if __name__ == "__main__":
    main()