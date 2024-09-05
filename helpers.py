
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


def obtain_metrics(labels, probas):
    '''
    Calculate performance on various metrics

    Args: 
        labels (np.array): labels of samples, should be True/False
        probas (np.array): predicted probabilities of samples, should be in [0, 1]
            and should be generated with predict_proba()[:, 1]
    Returns:
        (list): [ Precision, Recall, Accuracy, F-Measure, AUC, MCC, Brier Score ]
    '''
    ret = []
    preds = probas > 0.5
    auc = metrics.roc_auc_score(labels, probas)
    ret.append(metrics.precision_score(labels, preds))
    ret.append(metrics.recall_score(labels, preds))
    ret.append(metrics.accuracy_score(labels, preds))
    ret.append(metrics.f1_score(labels, preds))
    ret.append(np.max(auc, 1.0 - auc))
    ret.append(metrics.matthews_corrcoef(labels, preds))
    ret.append(metrics.brier_score_loss(labels, probas))

    return ret

def obtain_period_data(dataset):
    features, labels = obtain_data(dataset, 'm')
    terminals = obtain_intervals(dataset)
    feature_list = []
    label_list = []

    for i in range(len(terminals) - 1):
        idx = np.logical_and(features[:, 0] >= terminals[i], features[:, 0] < terminals[i + 1])
        feature_list.append(features[idx][:, 1:])
        label_list.append(labels[idx])
    return feature_list, label_list


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
        end_time = start_time + 28 * unit_period
    elif dataset == 'b':
        # time unit in Backblaze: month, tracing time: one year (12 months)
        start_time = 1
        unit_period = 1  # unit period: one month
        end_time = start_time + 12 * unit_period
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
    idx_true = np.where(training_labels == True)[0]
    idx_false = np.where(training_labels == False)[0]
    idx_false_resampled = resample(idx_false, n_samples=len(idx_true)*ratio, replace=False, random_state = random_seed)
    idx_resampled = np.concatenate([idx_false_resampled, idx_true])
    idx_resampled.sort()
    resampled_features = training_features[idx_resampled]
    resampled_labels = training_labels[idx_resampled]
    return resampled_features, resampled_labels

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
    elif(dataset=='a'):
        df = pd.read_csv(DATASET_PATH)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        feature_list, label_list = obtain_period_data('a')    
    else:
        print('Incorrect Dataset')
    return feature_list, label_list

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
    v, p = stat_test(distribution_reference, distribution_test)
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
    ks_test_time_start = time.time()
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
