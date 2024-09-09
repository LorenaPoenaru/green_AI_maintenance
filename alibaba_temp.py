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

from helpers import *





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
    training_features_init = np.vstack(feature_list[0: i])
    training_labels_init = np.hstack(label_list[0//2: i])

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
        testing_features = feature_list[i]
        testing_labels = label_list[i]
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

        partial_roc_auc_ks_all_model.append(roc_auc_score(testing_labels_model, predictions_test_updated))
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

    # obtain training features and labels
    training_features_init = np.vstack(feature_list[0: i])
    training_labels_init = np.hstack(label_list[0//2: i])


    for batch in tqdm(range(num_chunks//2, num_chunks)):
        
        # init drift alert
        drift_alert = 0

        # check if it is the first batch
        if(batch==num_chunks//2):
            training_features = training_features_init
            training_labels = training_labels_init
            current_training_batches_list = initial_training_batches_list.copy()
            print('Initial Training Batches', current_training_batches_list)

        

        # scaler on training data (No downsampling)
        update_scaler = StandardScaler()
        training_features = update_scaler.fit_transform(training_features)
        training_labels = training_labels

        
        # obtain testing features and labels
        testing_features = feature_list[i]
        testing_labels = label_list[i]

        
        # scaling testing features
        testing_features = update_scaler.transform(testing_features)
        testing_labels = testing_labels


        # training model
        begin_train_fh_ks_pca = time.time()


        if(batch==num_chunks//2 or need_to_retrain == 1):
            print('RETRAINING MODEL')
            
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
        
        
        # evaluate model on testing data
        
        begin_test_time_ks_pca = time.time()
        
        predictions_test_updated, total_testing_tracker_values = get_predictions(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, testing_features, total_testing_tracker_values, tracker)
        
        end_test_time_ks_pca = time.time() - begin_test_time_ks_pca
        
        total_test_time_ks_pca = total_test_time_ks_pca + end_test_time_ks_pca
        
        
        # ROC AUC
        partial_roc_auc_ks_pca_model.append(roc_auc_score(testing_labels, predictions_test_updated))
        predictions_test_ks_pca_model = np.concatenate([predictions_test_ks_pca_model, predictions_test_updated]) 
        
        print('Predictions Test Batch', len(predictions_test_updated))
        print('Prediction Test All', len(predictions_test_ks_pca_model))
        
        
        # Drift Detection
        need_to_retrain = 0
        drift_time_start = time.time()
        
        # Extract PCA Features
        pca_computing_time_start = time.time()
        df_train_features_sorted_pca, df_test_features_sorted_pca, total_pca_tracker_values =run_pca(dataset_name, type_retraining_data, detection, random_seed, batch, training_features, testing_features, total_pca_tracker_values, tracker)
        pca_computing_time_end = time.time() - pca_computing_time_start
        
        # Detect Drift
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
    training_features_init = np.vstack(feature_list[0: i])
    training_labels_init = np.hstack(label_list[0//2: i])

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
        testing_features = feature_list[i]
        testing_labels = label_list[i]
        
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
            
            training_features_list_updated = [feature_list[i] for i in current_training_batches_list]
            training_labels_list_updated = [label_list[i] for i in current_training_batches_list]
        
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

    configurations =  [("FullHistory","KS-ALL"), ("FullHistory", "KS-PCA"), ("FullHistory", "KS-FI")]
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