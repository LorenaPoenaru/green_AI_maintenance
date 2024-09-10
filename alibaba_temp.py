from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn import metrics, preprocessing
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import seaborn as sns
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import random
import time
from sklearn.model_selection import RandomizedSearchCV

#Energy Measurement Tool
from codecarbon import EmissionsTracker

from utilities import obtain_period_data, obtain_metrics

def set_name_tracker_for_task(dataset_name, type_retraining_data, detection, task, random_seed, batch):
    return str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(task) + "_RandomSeed_" + str(random_seed) + "_Batch" + str(batch)

def initiate_tracker_var():
    return {'cpu': 0, 'gpu': 0, 'ram':0, 'duration':0}

def initiate_tracker_variables():    
    total_hyperparam_tracker_values = initiate_tracker_var()
    total_fit_tracker_values = initiate_tracker_var()
    total_testing_tracker_values = initiate_tracker_var()
    return total_hyperparam_tracker_values, total_fit_tracker_values, total_testing_tracker_values

def format_data_for_the_seed(columns_names, values):    
    df_results_periodic_fh = pd.DataFrame(columns=columns_names)
    df_results_periodic_fh.loc[0] = values
    return df_results_periodic_fh

def store_into_file(filename, df_results_periodic_fh):
    df_results_periodic_fh.to_csv(filename, index=False,  sep=";") 

def features_labels_preprocessing(DATASET_PATH, dataset):
    
    if(dataset=='b'):
        
        print('Data Reading and Preprocessing')
        
        # set data paths and columns names
        features_disk_failure = ['smart_1_raw', 'smart_4_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_12_raw', 'smart_187_raw', 'smart_193_raw', 'smart_194_raw', 'smart_197_raw', 'smart_199_raw', 
                         'smart_4_raw_diff', 'smart_5_raw_diff', 'smart_9_raw_diff', 'smart_12_raw_diff', 'smart_187_raw_diff', 'smart_193_raw_diff', 'smart_197_raw_diff', 'smart_199_raw_diff']
        columns = ['serial_number', 'date'] + features_disk_failure + ['label']
        
        # read dataset
        df = pd.read_csv(DATASET_PATH_DISK, header=None, dtype = 'str').iloc[1:,1:]
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

def ks_drift_detection(reference_data, testing_data, tracker, total_distribution_tracker_values, total_stats_tracker_values):
    
    # extract distributions from reference and testing data
    
    distribution_extraction_time_start = time.time()
    tracker.start_task(set_name_tracker_for_task(dataset_name, type_retraining_data, detection, "Distribution_Extraction", random_seed, i))   
    distribution_reference = sns.distplot(np.array(reference_data)).get_lines()[0].get_data()[1]
    plt.close()
    distribution_test = sns.distplot(np.array(testing_data)).get_lines()[0].get_data()[1]
    plt.close()
    distribution_emissions = tracker.stop_task()
    total_distribution_tracker_values['cpu'] += distribution_emissions.cpu_energy
    total_distribution_tracker_values['gpu'] += distribution_emissions.gpu_energy
    total_distribution_tracker_values['ram'] += distribution_emissions.ram_energy
    total_distribution_tracker_values['duration'] += distribution_emissions.duration
    distribution_extraction_time_end = time.time() - distribution_extraction_time_start
    # apply KS statistical test
    
    ks_test_time_start = time.time()
    tracker.start_task(set_name_tracker_for_task(dataset_name, type_retraining_data, detection, "KS_Stats", random_seed, i))   
    stat_test = stats.kstest
    v, p = stat_test(distribution_reference, distribution_test)
    stats_emissions = tracker.stop_task()
    total_stats_tracker_values['cpu'] += stats_emissions.cpu_energy
    total_stats_tracker_values['gpu'] += stats_emissions.gpu_energy
    total_stats_tracker_values['ram'] += stats_emissions.ram_energy
    total_stats_tracker_values['duration'] += stats_emissions.duration
    ks_test_time_end = time.time() - ks_test_time_start
    # check if drift
    
    if(p<0.05):
        drift_alert = 1
    else:
        drift_alert = 0

    return drift_alert, distribution_extraction_time_end, ks_test_time_end, total_distribution_tracker_values, total_stats_tracker_values



def pipeline_ks_all(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list):
    experiment_name = str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(random_seed) 

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

    # obtain training features and labels
    training_features_init = np.vstack(feature_list[0: num_chunks//2])
    training_labels_init = np.hstack(label_list[0//2: num_chunks//2])

    for batch in tqdm(range(num_chunks//2, num_chunks)):
        # init drift alert
        drift_alert = 0
        
        # check if it is the first batch
        if(batch==num_chunks//2):
            training_features = training_features_init
            training_labels = training_labels_init
            current_training_batches_list = initial_training_batches_list.copy()
            print('Initial Training Batches', current_training_batches_list)

        # scaler for training data
        update_scaler = StandardScaler()
        training_features = update_scaler.fit_transform(training_features)
        training_labels = training_labels
        # obtain testing features and labels
        testing_features = feature_list[batch]
        testing_labels = label_list[batch]
        # scaling testing features
        testing_features = update_scaler.transform(testing_features)
        testing_labels = testing_labels
        # training model        
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
        #predictions_test_updated = update_model_ks_all.predict(testing_features_model)
        
        end_test_time_ks_all = time.time() - begin_test_time_ks_all
        total_test_time_ks_all = total_test_time_ks_all + end_test_time_ks_all

        partial_roc_auc_ks_all_model.append(roc_auc_score(testing_labels, predictions_test_updated))
        predictions_test_ks_all_model = np.concatenate([predictions_test_ks_all_model, predictions_test_updated])
        
        
        print('Predictions Test Batch', len(predictions_test_updated))
        print('Prediction Test All', len(predictions_test_ks_all_model))
        
        
        # Drift Detection
        
        need_to_retrain = 0
        
        print('MODEL', update_model_ks_all)
        
        
        drift_time_start = time.time()
        #drift_alert, distribution_extraction_time, ks_test_time = ks_drift_detection(training_features, testing_features)
        drift_alert, distribution_extraction_time, ks_test_time, total_distribution_tracker_values, total_stats_tracker_values = ks_drift_detection(dataset_name, type_retraining_data, detection, random_seed, i, training_features, testing_features, total_distribution_tracker_values, total_stats_tracker_values, tracker)
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

        print('Current Training Batches',current_training_batches_list)
    
    columns_names = ['Random_Seed', 'Model', 'Drifts_Overall',  
    'ROC_AUC_Batch', 'ROC_AUC_BATCH_MEAN', 'ROC_AUC_Total', 'Predictions', 
    'True_Testing_Labels', 'Train_Time', 'Hyperparam_Tunning_Time', 'Test_Time', 
    'Drifts_Detected', 'Drift_Detection_Total_Time', 'Distribution_Extraction_Time', 
    'Statistical_Test_Time', 'Label_Costs',
    'CPU_Energy_Hyperparameter', 'GPU_Energy_Hyperparameter', 'RAM_Energy_Hyperparameter', 'Duration_Tracker_Hyperparameter', 
    'CPU_Energy_Fitting', 'GPU_Energy_Fitting', 'RAM_Energy_Fitting', 'Duration_Tracker_Fitting', 
    'CPU_Energy_Testing', 'GPU_Energy_Testing', 'RAM_Energy_Testing', 'Duration_Tracker_Testing',
    'CPU_Energy_Distribution_Extraction', 'GPU_Energy_Distribution_Extraction', 'RAM_Energy_Distribution_Extraction', 'Duration_Tracker_Distribution_Extraction',
    'CPU_Energy_Stats_Test', 'GPU_Energy_Stats_Test', 'RAM_Energy_Stats_Test', 'Duration_Tracker_Stats_Test'
    ]
    values = [random_seed, 'KS_ALL', str(no_necessary_retrainings)+'/'+str(len(detected_drifts)), 
    partial_roc_auc_ks_all_model, np.mean(partial_roc_auc_ks_all_model), 
    roc_auc_score(true_testing_labels, predictions_test_ks_all_model), predictions_test_ks_all_model, 
    true_testing_labels, total_train_fh_all, total_hyperparam_fh_ks_all, total_test_time_ks_all, 
    detected_drifts, total_drift_detection_time, total_distribution_extraction_time, total_stat_test_time, 
    necessary_label_annotation_effort,
    total_hyperparam_tracker_values['cpu'], total_hyperparam_tracker_values['gpu'], total_hyperparam_tracker_values['ram'], total_hyperparam_tracker_values['duration'], 
    total_fit_tracker_values['cpu'], total_fit_tracker_values['gpu'], total_fit_tracker_values['ram'], total_fit_tracker_values['duration'], 
    total_testing_tracker_values['cpu'], total_testing_tracker_values['gpu'], total_testing_tracker_values['ram'], total_testing_tracker_values['duration'],
    total_distribution_tracker_values['cpu'], total_distribution_tracker_values['gpu'], total_distribution_tracker_values['ram'], total_distribution_tracker_values['duration'],
    total_stats_tracker_values['cpu'], total_stats_tracker_values['gpu'], total_stats_tracker_values['ram'], total_stats_tracker_values['duration']]

    df_results_for_seed = format_data_for_the_seed(columns_names, values)
    store_into_file('./results/Output_' + str(experiment_name) + ".csv", df_results_for_seed)
    _ = tracker.stop()


def pipeline_ks_pca(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list):
    experiment_name = str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(random_seed) 
       
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

    for i in tqdm(range(num_chunks//2, num_chunks)):
        # obtain training features and labels
        training_features_init = np.vstack(feature_list[0: i])
        training_labels_init = np.hstack(label_list[0//2: i])
        
        # init drift alert
        drift_alert = 0

        # check if it is the first batch
        if(i==num_chunks//2):
            training_features = training_features_init
            training_labels = training_labels_init
            current_training_batches_list = initial_training_batches_list.copy()
            print('Initial Training Batches', current_training_batches_list)

            # obtain testing features and labels
        testing_features = feature_list[i]
        testing_labels = label_list[i]

        # scaler on training data (No downsampling)
        update_scaler = StandardScaler()
        training_features_model = update_scaler.fit_transform(training_features)
        training_labels_model = training_labels

        # scaling testing features
        testing_features_model = update_scaler.transform(testing_features)
        testing_labels_model = testing_labels

        # training model
        begin_train_fh_ks_pca = time.time()


        if(i==num_chunks//2 or need_to_retrain == 1):
            print('RETRAINING MODEL')
            
            begin_train_fh_ks_pca = time.time()
        
            begin_hyperparam_tunning_update = time.time()
            tracker.start_task(set_name_tracker_for_task(dataset_name, type_retraining_data, detection, "HyperparameterTuning", random_seed, i))
            model = RandomForestClassifier(random_state = random_seed)
            random_search = RandomizedSearchCV(model,
                                                       param_distributions = param_dist_rf,
                                                       n_iter=N_ITER_SEARCH,
                                                       scoring='roc_auc',
                                                       cv=4, n_jobs=1, random_state = random_seed)

            
            random_search.fit(training_features_model, training_labels_model)
            update_model_ks_pca = random_search.best_estimator_
            hyperparameter_tuning_emissions = tracker.stop_task()
            total_hyperparam_tracker_values['cpu'] += hyperparameter_tuning_emissions.cpu_energy
            total_hyperparam_tracker_values['gpu'] += hyperparameter_tuning_emissions.gpu_energy
            total_hyperparam_tracker_values['ram'] += hyperparameter_tuning_emissions.ram_energy
            total_hyperparam_tracker_values['duration'] += hyperparameter_tuning_emissions.duration

            end_hyperparam_tunning_update = time.time() - begin_hyperparam_tunning_update
            total_hyperparam_fh_ks_pca = total_hyperparam_fh_ks_pca + end_hyperparam_tunning_update
            
            
            
            tracker.start_task(set_name_tracker_for_task(dataset_name, type_retraining_data, detection, "ModelFit", random_seed, i))
            update_model_ks_pca.fit(training_features_model, training_labels_model)
            model_fit_emissions = tracker.stop_task()
            total_fit_tracker_values['cpu'] += model_fit_emissions.cpu_energy
            total_fit_tracker_values['gpu'] += model_fit_emissions.gpu_energy
            total_fit_tracker_values['ram'] += model_fit_emissions.ram_energy
            total_fit_tracker_values['duration'] += model_fit_emissions.duration
            end_train_fh_ks_pca = time.time() - begin_train_fh_ks_pca
        
            total_train_fh_pca = total_train_fh_pca + end_train_fh_ks_pca
        
        
        # evaluate model on testing data
        
        begin_test_time_ks_pca = time.time()
        tracker.start_task(set_name_tracker_for_task(dataset_name, type_retraining_data, detection, "Testing", random_seed, i))
        predictions_test_updated = update_model_ks_pca.predict(testing_features_model)
        testing_emissions = tracker.stop_task()
        total_testing_tracker_values['cpu'] += testing_emissions.cpu_energy
        total_testing_tracker_values['gpu'] += testing_emissions.gpu_energy
        total_testing_tracker_values['ram'] += testing_emissions.ram_energy
        total_testing_tracker_values['duration'] += testing_emissions.duration
        end_test_time_ks_pca = time.time() - begin_test_time_ks_pca
        total_test_time_ks_pca = total_test_time_ks_pca + end_test_time_ks_pca
        
        
        # ROC AUC
        partial_roc_auc_ks_pca_model.append(roc_auc_score(testing_labels_model, predictions_test_updated))
        predictions_test_ks_pca_model = np.concatenate([predictions_test_ks_pca_model, predictions_test_updated])
        print('Predictions Test Batch', len(predictions_test_updated))
        print('Prediction Test All', len(predictions_test_ks_pca_model))
        
        # Drift Detection
        need_to_retrain = 0
        print('MODEL', update_model_ks_pca)
        
        drift_time_start = time.time()
        
        # Extract PCA Features
        pca_computing_time_start = time.time()
        tracker.start_task(set_name_tracker_for_task(dataset_name, type_retraining_data, detection, "PCA", random_seed, i))
        pca = PCA(n_components = 0.95, random_state = random_seed)
        pca.fit(training_features_model)
        df_train_features_sorted_pca = pca.transform(training_features_model)
        df_test_features_sorted_pca = pca.transform(testing_features_model)
        pca_emissions = tracker.stop_task()
        total_pca_tracker_values['cpu'] += pca_emissions.cpu_energy
        total_pca_tracker_values['gpu'] += pca_emissions.gpu_energy
        total_pca_tracker_values['ram'] += pca_emissions.ram_energy
        total_pca_tracker_values['duration'] += pca_emissions.duration
        pca_computing_time_end = time.time() - pca_computing_time_start
        
        # Detect Drift
        drift_alert, distribution_extraction_time, ks_test_time, total_distribution_tracker_values, total_stats_tracker_values = ks_drift_detection(df_train_features_sorted_pca, df_test_features_sorted_pca, tracker, total_distribution_tracker_values, total_stats_tracker_values)
        drift_time_end = time.time() - drift_time_start

        total_distribution_extraction_time = total_distribution_extraction_time + distribution_extraction_time
        total_stat_test_time = total_stat_test_time + ks_test_time
        total_pca_time = total_pca_time + pca_computing_time_end
        total_drift_detection_time = total_drift_detection_time + drift_time_end
        
        detected_drifts.append(drift_alert)
        
        if(drift_alert==1):
            need_to_retrain = 1
            drift_alert = 0
            print('CHANGE OF TRAINING')
            no_necessary_retrainings = no_necessary_retrainings + 1
            necessary_label_annotation_effort = necessary_label_annotation_effort + len(testing_labels)

            
            # add new data to the training for full history approach
            current_training_batches_list.append(i)
                    
            
            training_features_list_updated = [feature_list[i] for i in current_training_batches_list]
            training_labels_list_updated = [label_list[i] for i in current_training_batches_list]
        
            training_features = np.vstack(training_features_list_updated)
            training_labels = np.hstack(training_labels_list_updated)   
        
        print('Current Training Batches',current_training_batches_list)
    
    
    columns_names = ['Random_Seed', 'Model', 'Drifts_Overall',  'ROC_AUC_Batch', 
    'ROC_AUC_BATCH_MEAN', 'ROC_AUC_Total', 'Predictions', 'True_Testing_Labels', 
    'Train_Time', 'Hyperparam_Tunning_Time', 'Test_Time', 'Drifts_Detected', 
    'Drift_Detection_Total_Time', 'PCA_Computing_time', 'Distribution_Extraction_Time', 
    'Statistical_Test_Time', 'Label_Costs',
    'CPU_Energy_Hyperparameter', 'GPU_Energy_Hyperparameter', 'RAM_Energy_Hyperparameter', 'Duration_Tracker_Hyperparameter', 
    'CPU_Energy_Fitting', 'GPU_Energy_Fitting', 'RAM_Energy_Fitting', 'Duration_Tracker_Fitting', 
    'CPU_Energy_Testing', 'GPU_Energy_Testing', 'RAM_Energy_Testing', 'Duration_Tracker_Testing',
    'CPU_Energy_Distribution_Extraction', 'GPU_Energy_Distribution_Extraction', 'RAM_Energy_Distribution_Extraction', 'Duration_Tracker_Distribution_Extraction',
    'CPU_Energy_Stats_Test', 'GPU_Energy_Stats_Test', 'RAM_Energy_Stats_Test', 'Duration_Tracker_Stats_Test',
    'CPU_Energy_PCA', 'GPU_Energy_PCA', 'RAM_Energy_PCA', 'Duration_Tracker_PCA']
    values = [random_seed, 'KS_PCA', str(no_necessary_retrainings)+'/'+str(len(detected_drifts)), 
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
    store_into_file('./results/Output_' + str(experiment_name) + ".csv", df_results_for_seed)
    _ = tracker.stop()
    




def pipeline_ks_fi(features_disk_failure, dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list):
    experiment_name = str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(random_seed) 

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

    # obtain training features and labels   
    training_features_init = np.vstack(feature_list[0: num_chunks//2])
    training_labels_init = np.hstack(label_list[0//2: num_chunks//2])

    for batch in tqdm(range(num_chunks//2, num_chunks)):
        
        # init drift alert
        drift_alert = 0

        # check if it is the first batch
        if(batch==num_chunks//2):
            training_features = training_features_init
            training_labels = training_labels_init
            current_training_batches_list = initial_training_batches_list.copy()
            print('Initial Training Batches', current_training_batches_list)
        
        # scaler training data (NO downsampling)
        update_scaler = StandardScaler()
        training_features = update_scaler.fit_transform(training_features)
        training_labels = training_labels
        
        # obtain testing features and labels
        testing_features = feature_list[batch]
        testing_labels = label_list[batch]
        
        # scaling testing features
        testing_features = update_scaler.transform(testing_features)
        testing_labels = testing_labels
        
        # training model
        begin_train_fh_ks_fi = time.time()

        if(batch==num_chunks//2 or need_to_retrain == 1):
            print('RETRAINING MODEL')
            
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
        
        
        # evaluate model on testing data
        
        begin_test_time_ks_fi = time.time()
        predictions_test_updated, total_testing_tracker_values = get_predictions(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, testing_features, total_testing_tracker_values, tracker)
        end_test_time_ks_fi = time.time() - begin_test_time_ks_fi
        total_test_time_ks_fi = total_test_time_ks_fi + end_test_time_ks_fi
        
        # ROC AUC & Predictions
        partial_roc_auc_ks_fi_model.append(roc_auc_score(testing_labels, predictions_test_updated))
        predictions_test_ks_fi_model = np.concatenate([predictions_test_ks_fi_model, predictions_test_updated])
    
        # Drift Detection
        need_to_retrain = 0
        
        drift_time_start = time.time()
        # Extract Most Important Features
        feature_importance_extraction_start = time.time()
        
        important_features, training_important_features_model, testing_important_features_model, total_fi_tracker_values = get_fi(dataset_name, type_retraining_data, detection, random_seed, batch, tracker, total_fi_tracker_values, update_model, training_features, testing_features, features_disk_failure)
        feature_importance_extraction_end = time.time() - feature_importance_extraction_start
        
        
        # Detect Drift
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
            
            training_features_list_updated = [feature_list[batch] for i in current_training_batches_list]
            training_labels_list_updated = [label_list[batch] for i in current_training_batches_list]
        
            training_features = np.vstack(training_features_list_updated)
            training_labels = np.hstack(training_labels_list_updated)
            
        
        print('Current Training Batches',current_training_batches_list)
    
    
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
    values = [random_seed, 'KS_FI', str(no_necessary_retrainings)+'/'+str(len(detected_drifts)), 
    partial_roc_auc_ks_fi_model, np.mean(partial_roc_auc_ks_fi_model), 
    roc_auc_score(true_testing_labels, predictions_test_ks_fi_model), predictions_test_ks_fi_model, 
    true_testing_labels, total_train_fh_fi, total_hyperparam_fh_ks_fi, total_test_time_ks_fi, 
    detected_drifts, total_drift_detection_time, total_feature_importance_extraction_time, 
    total_distribution_extraction_time, total_stat_test_time, necessary_label_annotation_effort,
    total_hyperparam_tracker_values['cpu'], total_hyperparam_tracker_values['gpu'], total_hyperparam_tracker_values['ram'], total_hyperparam_tracker_values['duration'],    
    total_fit_tracker_values['cpu'], total_fit_tracker_values['gpu'], total_fit_tracker_values['ram'], total_fit_tracker_values['duration'], 
    total_testing_tracker_values['cpu'], total_testing_tracker_values['gpu'], total_testing_tracker_values['ram'], total_testing_tracker_values['duration'],
    total_distribution_tracker_values['cpu'], total_distribution_tracker_values['gpu'], total_distribution_tracker_values['ram'], total_distribution_tracker_values['duration'],
    total_stats_tracker_values['cpu'], total_stats_tracker_values['gpu'], total_stats_tracker_values['ram'], total_stats_tracker_values['duration'],
    total_fi_tracker_values['cpu'], total_fi_tracker_values['gpu'], total_fi_tracker_values['ram'], total_fi_tracker_values['duration']]

    df_results_for_seed = format_data_for_the_seed(columns_names, values)
    store_into_file('./results/Output_' + str(experiment_name) + ".csv", df_results_for_seed)
    _ = tracker.stop()


def main():
    DATASET_PATH = 'alibaba_job_data.csv'
    feature_list, label_list = features_labels_preprocessing(DATASET_PATH, 'a')
    num_chunks = len(feature_list)
    true_testing_labels = np.hstack(label_list[num_chunks//2:])
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

    features_failure = ['user', 'task_name', 'inst_num', 'plan_cpu', 'plan_mem', 'plan_gpu', 
    'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem']
    initial_training_batches_list = list(range(0, num_chunks//2))
    
    N_WORKERS = 1
    TOTAL_NUMBER_SEEDS = 1
    random_seeds = list(np.arange(TOTAL_NUMBER_SEEDS))
    N_ITER_SEARCH = 100
    dataset_name = "Alibaba"
    print(DATASET_PATH)

    configurations =  [("FullHistory", "KS-PCA")] #, ("FullHistory", "KS-FI")] #("FullHistory","KS-ALL"), 
    counter = {}
    for configuration in configurations:
        counter[configuration] = TOTAL_NUMBER_SEEDS
    executions = configurations * TOTAL_NUMBER_SEEDS
    random.shuffle(executions)
    print(executions)
    for configuration in tqdm(executions):
        print(configuration)
        type_retraining_data = configuration[0]
        detection = configuration[1]
        random_seed = random_seeds[counter[configuration]-1]
        if detection == "KS-ALL":
            pipeline_ks_all(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
        if detection == "KS-PCA":
            pipeline_ks_pca(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
        if detection == "KS-FI":
            pipeline_ks_fi(features_failure, dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
    print("End of Experimentation")


if __name__ == "__main__":
    main()