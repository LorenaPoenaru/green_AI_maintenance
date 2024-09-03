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


from helpers import *

def get_scaled_transformation_testing_features(scaler, testing_features):
    return scaler.transform(testing_features)



def pipeline_periodic_model(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list):
    experiment_name = str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(random_seed) + ".csv"
    
    total_time_training = 0
    predictions_test_fh = []
    lengths_training_fh = []
    partial_roc_auc_fh = []
    total_train_fh = 0 
    total_hyperparam_fh = 0
    total_test_fh = 0
    
    ### Tracker tasks variables 
    tracker = EmissionsTracker(project_name="AI_maintenance_" + str(experiment_name))
    total_hyperparam_tracker_values, total_fit_tracker_values, total_testing_tracker_values = initiate_tracker_variables()
    training_features, training_labels = initiate_training_features_and_labels_from_lists(feature_list, label_list, num_chunks)
    begin_total_fh = time.time()
    for batch in tqdm(range(num_chunks//2, num_chunks)):
        # obtain features and labels
        testing_features, testing_labels = get_testing_features_and_labels_from_lists(feature_list, label_list, batch)
        # scaling data
        training_features, testing_features = scaling_data(training_features, testing_features)
        # Downscaling for data training
        training_features, training_labels = downsampling(training_features, training_labels, random_seed)
        # training model
        begin_train_fh = time.time()
        # Hyperparameter tunning energy collection start
        begin_hyperparam_tunning_update = time.time()
        update_model, total_hyperparam_tracker_values = hyperparameter_tuning_process(dataset_name, type_retraining_data, 
                                                                                          detection, random_seed, batch, param_dist_rf, 
                                                                                          N_ITER_SEARCH, training_features, 
                                                                                          training_labels,
                                                                                          total_hyperparam_tracker_values, tracker)
        end_hyperparam_tunning_update = time.time() - begin_hyperparam_tunning_update
        # Hyperparameter tunning energy collection end
        update_model, total_fit_tracker_values = best_model_fit(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, training_features, training_labels, total_fit_tracker_values, tracker)
        # Fitting the model energy collection end
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
        lengths_training_fh.append(len(training_features))
        training_labels = np.hstack(label_list[0: batch+1])

    end_total_fh = time.time() - begin_total_fh
    
    
    columns_names =['Random_Seed', 'Model', 'Drifts', 'ROC_AUC_Batch', 'ROC_AUC_BATCH_MEAN', 'ROC_AUC_Total', 
    'Predictions', 'True_Testing_Labels', 'Train_Time', 'Hyperparam_Tunning_Time', 'Test_Time', 'Drifts_Detected', 'Label_Costs',
    'CPU_Energy_Hyperparameter', 'GPU_Energy_Hyperparameter', 'RAM_Energy_Hyperparameter', 'Duration_Tracker_Hyperparameter', 
    'CPU_Energy_Fitting', 'GPU_Energy_Fitting', 'RAM_Energy_Fitting', 'Duration_Tracker_Fitting', 
    'CPU_Energy_Testing', 'GPU_Energy_Testing', 'RAM_Energy_Testing', 'Duration_Tracker_Testing']
    values =  [random_seed, 'periodic-sw', str(int(num_chunks//2)) + '/' + str(int(num_chunks//2)), 
    partial_roc_auc_fh, np.mean(partial_roc_auc_fh), roc_auc_score(true_testing_labels, predictions_test_fh), 
    predictions_test_fh, true_testing_labels,  total_train_fh, total_hyperparam_fh, total_test_fh, np.ones(int(num_chunks//2), dtype=int), 
    len(true_testing_labels),
    total_hyperparam_tracker_values['cpu'], total_hyperparam_tracker_values['gpu'], total_hyperparam_tracker_values['ram'], total_hyperparam_tracker_values['duration'], 
    total_fit_tracker_values['cpu'], total_fit_tracker_values['gpu'], total_fit_tracker_values['ram'], total_fit_tracker_values['duration'], 
    total_testing_tracker_values['cpu'], total_testing_tracker_values['gpu'], total_testing_tracker_values['ram'], total_testing_tracker_values['duration']
    ]
    df_results_for_seed = format_data_for_the_seed(columns_names, values)
    store_into_file('./results/Output_' + str(experiment_name), df_results_for_seed)
    _ = tracker.stop()


def pipeline_static_model(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list):
    experiment_name = str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(random_seed) + ".csv"

    begin_total_static = time.time()
    partial_roc_auc = []
    total_test_static = 0
    ### Tracker tasks variables 
    tracker = EmissionsTracker(project_name="AI_maintenance_" + str(experiment_name))
    total_hyperparam_tracker_values, total_fit_tracker_values, total_testing_tracker_values = initiate_tracker_variables()
    training_features, training_labels = initiate_training_features_and_labels_from_lists(feature_list, label_list, num_chunks)

    # scaling training data
    scaler = StandardScaler()
    training_features = scaler.fit_transform(training_features)

    # downsampling training data
    training_features, training_labels = downsampling(training_features, training_labels, random_seed)


    begin_train_time_static = time.time()
    begin_hyperparam_tunning_static = time.time()
    update_model, total_hyperparam_tracker_values = hyperparameter_tuning_process(dataset_name, type_retraining_data, 
                                                                                          detection, random_seed, "Static", param_dist_rf, 
                                                                                          N_ITER_SEARCH, training_features, 
                                                                                          training_labels,
                                                                                          total_hyperparam_tracker_values, tracker)
    end_hyperparam_tunning_static = time.time() - begin_hyperparam_tunning_static
    
    # Training
    update_model, total_fit_tracker_values = best_model_fit(dataset_name, type_retraining_data, detection, random_seed, "Static", update_model, training_features, training_labels, total_fit_tracker_values, tracker)
    end_train_time_static = time.time() - begin_train_time_static

    total_time_training = 0
    predictions_test_static_model = []
    # Testing model on periods
    begin_test_time_static = time.time()
    for batch in tqdm(range(num_chunks//2, num_chunks)):
        # obtain features and labels
        testing_features, testing_labels = get_testing_features_and_labels_from_lists(feature_list, label_list, batch)
        # scaling testing features
        testing_features = scaler.transform(testing_features)
        # evaluate model on testing data
        begin_test_time_static = time.time()
        predictions_test_updated, total_testing_tracker_values = get_predictions(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, testing_features, total_testing_tracker_values, tracker)
        end_test_time_static = time.time() - begin_test_time_static
        total_test_static = total_test_static + end_test_time_static

        partial_roc_auc.append(roc_auc_score(testing_labels, predictions_test_updated))
        predictions_test_static_model = np.concatenate([predictions_test_static_model, predictions_test_updated])

    end_total_static = time.time() - begin_total_static
    
    columns_names =['Random_Seed', 'Model', 'Drifts', 'ROC_AUC_Batch', 'ROC_AUC_BATCH_MEAN', 'ROC_AUC_Total', 'Predictions', 
    'True_Testing_Labels', 'Train_Time', 'Hyperparam_Tunning_Time', 'Test_Time', 'Drifts_Detected', 'Label_Costs',
    'CPU_Energy_Hyperparameter', 'GPU_Energy_Hyperparameter', 'RAM_Energy_Hyperparameter', 'Duration_Tracker_Hyperparameter', 
    'CPU_Energy_Fitting', 'GPU_Energy_Fitting', 'RAM_Energy_Fitting', 'Duration_Tracker_Fitting', 
    'CPU_Energy_Testing', 'GPU_Energy_Testing', 'RAM_Energy_Testing', 'Duration_Tracker_Testing']
    values = [random_seed, 'static', '0/' + str(int(num_chunks//2)), partial_roc_auc, np.mean(partial_roc_auc), 
    roc_auc_score(true_testing_labels, predictions_test_static_model), predictions_test_static_model, 
    true_testing_labels, end_train_time_static, end_hyperparam_tunning_static, total_test_static, 
    np.zeros(int(num_chunks//2), dtype=int), 0.0,
    total_hyperparam_tracker_values['cpu'], total_hyperparam_tracker_values['gpu'], total_hyperparam_tracker_values['ram'], total_hyperparam_tracker_values['duration'], 
    total_fit_tracker_values['cpu'], total_fit_tracker_values['gpu'], total_fit_tracker_values['ram'], total_fit_tracker_values['duration'], 
    total_testing_tracker_values['cpu'], total_testing_tracker_values['gpu'], total_testing_tracker_values['ram'], total_testing_tracker_values['duration']
    ]
    df_results_for_seed = format_data_for_the_seed(columns_names, values)
    store_into_file('./results/Output_' + str(experiment_name), df_results_for_seed)
    _ = tracker.stop()





def pipeline_static_model_debug(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list):
    experiment_name = str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(random_seed) + ".csv"

    begin_total_static = time.time()
    partial_roc_auc = []
    total_test_static = 0

    # extracting training features and labels
    training_features, training_labels = initiate_training_features_and_labels_from_lists(feature_list, label_list, num_chunks)

    # scaling training data
    scaler = StandardScaler()
    training_features = scaler.fit_transform(training_features)

    # downsampling training data
    training_features_downsampling, training_labels_downsampling = downsampling(training_features, training_labels, random_seed, ratio=10)

    # training model
    begin_train_time_static = time.time()
        
    begin_hyperparam_tunning_static = time.time()

    update_model, total_hyperparam_tracker_values = hyperparameter_tuning_process(dataset_name, type_retraining_data, detection, random_seed, batch, param_dist_rf, N_ITER_SEARCH, training_features_downsampling, training_labels_downsampling, total_hyperparam_tracker_values, tracker)
    
    end_hyperparam_tunning_static = time.time() - begin_hyperparam_tunning_static
    
    print('Training')
    update_model, total_fit_tracker_values = best_model_fit(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, training_features, training_labels, total_fit_tracker_values, tracker)
    end_train_time_static = time.time() - begin_train_time_static
    #print('Training time: ', end_train_time_static)

    total_time_training = 0
    predictions_test_static_model = []



    print('Testing model on periods')
    begin_test_time_static = time.time()
    for i in tqdm(range(num_chunks//2, num_chunks)):

        # obtain testing features and labels
        testing_features = feature_list[i]
        testing_labels = label_list[i]

        # scaling testing features
        testing_features = scaler.transform(testing_features)

        # evaluate model on testing data
        begin_test_time_static = time.time()
        predictions_test_updated = update_model.predict(testing_features)
        end_test_time_static = time.time() - begin_test_time_static
        total_test_static = total_test_static + end_test_time_static

        partial_roc_auc.append(roc_auc_score(testing_labels, predictions_test_updated))

        predictions_test_static_model = np.concatenate([predictions_test_static_model, predictions_test_updated])




    end_total_static = time.time() - begin_total_static
    
    df_results_static_rf = pd.DataFrame(columns=['Random_Seed', 'Model', 'Drifts', 'ROC_AUC_Batch', 'ROC_AUC_BATCH_MEAN', 'ROC_AUC_Total', 'Predictions', 'True_Testing_Labels', 'Train_Time', 'Hyperparam_Tunning_Time', 'Test_Time', 'Drifts_Detected', 'Label_Costs'])
    df_results_static_rf.loc[0] = [random_seed, 'static', '0/' + str(int(num_chunks//2)), partial_roc_auc, np.mean(partial_roc_auc), roc_auc_score(true_testing_labels, predictions_test_static_model), predictions_test_static_model, true_testing_labels, end_train_time_static, end_hyperparam_tunning_static, total_test_static, np.zeros(int(num_chunks//2), dtype=int), 0.0]

    df_results_disk = pd.concat([df_results_disk, df_results_static_rf])
    df_results_disk = df_results_disk.reset_index(drop=True)
    df_results_disk.to_csv('./results/static_model_backblaze_data_green.csv')



def pipeline_ks_all(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list):
    experiment_name = str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(random_seed) + ".csv"

    necessary_label_annotation_effort = 0
    total_time_training = 0
    no_necessary_retrainings = 0
    lengths_training_ks_all = []
    partial_roc_auc_ks_all_model = []
    predictions_test_ks_all_model = []
    total_train_all = 0
    total_hyperparam_ks_all = 0
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
        begin_train_ks_all = time.time()
        if(batch==num_chunks//2 or need_to_retrain == 1):
            begin_train_ks_all = time.time()
            begin_hyperparam_tunning_update = time.time()
            update_model, total_hyperparam_tracker_values = hyperparameter_tuning_process(dataset_name, type_retraining_data, 
                                                                                          detection, random_seed, batch, param_dist_rf, 
                                                                                          N_ITER_SEARCH, training_features, 
                                                                                          training_labels,
                                                                                          total_hyperparam_tracker_values, tracker)
            end_hyperparam_tunning_update = time.time() - begin_hyperparam_tunning_update
            total_hyperparam_ks_all = total_hyperparam_ks_all + end_hyperparam_tunning_update

            update_model, total_fit_tracker_values = best_model_fit(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, training_features, training_labels, total_fit_tracker_values, tracker)
            end_train_ks_all = time.time() - begin_train_ks_all
            total_train_all = total_train_all + end_train_ks_all
        
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

            # remove data to the training for sliding window approach
            current_training_batches_list.remove(current_training_batches_list[0])
            current_training_batches_list.append(batch)
            training_features_list_updated = [feature_list[i] for i in current_training_batches_list]
            training_labels_list_updated = [label_list[i] for i in current_training_batches_list]
            training_features = np.vstack(training_features_list_updated)
            training_labels = np.hstack(training_labels_list_updated)

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
              predictions_test_ks_all_model, true_testing_labels, total_train_all, total_hyperparam_ks_all, 
              total_test_time_ks_all, detected_drifts, total_drift_detection_time, total_distribution_extraction_time, 
              total_stat_test_time, necessary_label_annotation_effort,
              total_hyperparam_tracker_values['cpu'], total_hyperparam_tracker_values['gpu'], total_hyperparam_tracker_values['ram'], total_hyperparam_tracker_values['duration'], 
              total_fit_tracker_values['cpu'], total_fit_tracker_values['gpu'], total_fit_tracker_values['ram'], total_fit_tracker_values['duration'], 
              total_testing_tracker_values['cpu'], total_testing_tracker_values['gpu'], total_testing_tracker_values['ram'], total_testing_tracker_values['duration'],
              total_distribution_tracker_values['cpu'], total_distribution_tracker_values['gpu'], total_distribution_tracker_values['ram'], total_distribution_tracker_values['duration'],
              total_stats_tracker_values['cpu'], total_stats_tracker_values['gpu'], total_stats_tracker_values['ram'], total_stats_tracker_values['duration']]
    
    df_results_for_seed = format_data_for_the_seed(columns_names, values)
    store_into_file('./results/Output_' + str(experiment_name), df_results_for_seed)
    _ = tracker.stop()



def pipeline_ks_pca(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list):
    experiment_name = str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(random_seed) + ".csv"

    necessary_label_annotation_effort = 0
    no_necessary_retrainings = 0
    partial_roc_auc_ks_pca_model = []
    predictions_test_ks_pca_model = []
    total_train_pca = 0
    total_hyperparam_ks_pca = 0
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
    
        begin_train_ks_pca = time.time()

        if(batch==num_chunks//2 or need_to_retrain == 1):
            begin_train_ks_pca = time.time()
            begin_hyperparam_tunning_update = time.time()
            update_model, total_hyperparam_tracker_values = hyperparameter_tuning_process(dataset_name, type_retraining_data, 
                                                                                          detection, random_seed, batch, param_dist_rf, 
                                                                                          N_ITER_SEARCH, training_features, 
                                                                                          training_labels,
                                                                                          total_hyperparam_tracker_values, tracker)
            end_hyperparam_tunning_update = time.time() - begin_hyperparam_tunning_update
            total_hyperparam_ks_pca = total_hyperparam_ks_pca + end_hyperparam_tunning_update
            
            update_model, total_fit_tracker_values = best_model_fit(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, training_features, training_labels, total_fit_tracker_values, tracker)
            end_train_ks_pca = time.time() - begin_train_ks_pca
            total_train_pca = total_train_pca + end_train_ks_pca
        
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

            # remove the old data and add new one for sliding window approach
            current_training_batches_list.remove(current_training_batches_list[0])
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
    predictions_test_ks_pca_model, true_testing_labels, total_train_pca, 
    total_hyperparam_ks_pca, total_test_time_ks_pca, detected_drifts, 
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
    _ = tracker.stop()



def pipeline_ks_fi(features_disk_failure, dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list):
    experiment_name = str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(random_seed) + ".csv"

    no_necessary_retrainings = 0
    necessary_label_annotation_effort = 0
    partial_roc_auc_ks_fi_model = []    
    predictions_test_ks_fi_model = []
    total_train_fi = 0
    total_hyperparam_ks_fi = 0
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
        begin_train_ks_fi = time.time()


        if(batch==num_chunks//2 or need_to_retrain == 1):
 
            begin_train_ks_fi = time.time()
            begin_hyperparam_tunning_update = time.time()
            update_model, total_hyperparam_tracker_values = hyperparameter_tuning_process(dataset_name, type_retraining_data, 
                                                                                          detection, random_seed, batch, param_dist_rf, 
                                                                                          N_ITER_SEARCH, training_features, 
                                                                                          training_labels,
                                                                                          total_hyperparam_tracker_values, tracker)
            end_hyperparam_tunning_update = time.time() - begin_hyperparam_tunning_update
            total_hyperparam_ks_fi = total_hyperparam_ks_fi + end_hyperparam_tunning_update
            
            update_model, total_fit_tracker_values = best_model_fit(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, training_features, training_labels, total_fit_tracker_values, tracker)
            end_train_ks_fi = time.time() - begin_train_ks_fi
            total_train_fi = total_train_fi + end_train_ks_fi
        
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
            
            # remove oldest data and add newest for sliding window approach
            current_training_batches_list.remove(current_training_batches_list[0]) 
            current_training_batches_list.append(batch)
            
            training_features_list_updated = [feature_list[i] for i in current_training_batches_list]
            training_labels_list_updated = [label_list[i] for i in current_training_batches_list]
        
            training_features = np.vstack(training_features_list_updated)
            training_labels = np.hstack(training_labels_list_updated)


    
    
    columns_names =['Random_Seed', 'Model', 'Drifts_Overall',  'ROC_AUC_Batch', 'ROC_AUC_BATCH_MEAN', 'ROC_AUC_Total', 'Predictions', 
    'True_Testing_Labels', 'Train_Time', 'Hyperparam_Tunning_Time', 'Test_Time', 'Drifts_Detected', 'Drift_Detection_Total_Time', 
    'FI_Extraction_Time', 'Distribution_Extraction_Time', 'Statistical_Test_Time', 'Label_Costs',
    'CPU_Energy_Hyperparameter', 'GPU_Energy_Hyperparameter', 'RAM_Energy_Hyperparameter', 'Duration_Tracker_Hyperparameter', 
    'CPU_Energy_Fitting', 'GPU_Energy_Fitting', 'RAM_Energy_Fitting', 'Duration_Tracker_Fitting', 
    'CPU_Energy_Testing', 'GPU_Energy_Testing', 'RAM_Energy_Testing', 'Duration_Tracker_Testing',
    'CPU_Energy_Distribution_Extraction', 'GPU_Energy_Distribution_Extraction', 'RAM_Energy_Distribution_Extraction', 'Duration_Tracker_Distribution_Extraction',
    'CPU_Energy_Stats_Test', 'GPU_Energy_Stats_Test', 'RAM_Energy_Stats_Test', 'Duration_Tracker_Stats_Test',
    'CPU_Energy_FI', 'GPU_Energy_FI', 'RAM_Energy_FI', 'Duration_Tracker_FI']
    values= [random_seed, 'KS_FI', str(no_necessary_retrainings)+'/'+str(len(detected_drifts)), partial_roc_auc_ks_fi_model, np.mean(partial_roc_auc_ks_fi_model), 
    roc_auc_score(true_testing_labels, predictions_test_ks_fi_model), predictions_test_ks_fi_model, true_testing_labels, 
    total_train_fi, total_hyperparam_ks_fi, total_test_time_ks_fi, detected_drifts, total_drift_detection_time, total_feature_importance_extraction_time, 
    total_distribution_extraction_time, total_stat_test_time, necessary_label_annotation_effort,
    total_hyperparam_tracker_values['cpu'], total_hyperparam_tracker_values['gpu'], total_hyperparam_tracker_values['ram'], total_hyperparam_tracker_values['duration'],    
    total_fit_tracker_values['cpu'], total_fit_tracker_values['gpu'], total_fit_tracker_values['ram'], total_fit_tracker_values['duration'], 
    total_testing_tracker_values['cpu'], total_testing_tracker_values['gpu'], total_testing_tracker_values['ram'], total_testing_tracker_values['duration'],
    total_distribution_tracker_values['cpu'], total_distribution_tracker_values['gpu'], total_distribution_tracker_values['ram'], total_distribution_tracker_values['duration'],
    total_stats_tracker_values['cpu'], total_stats_tracker_values['gpu'], total_stats_tracker_values['ram'], total_stats_tracker_values['duration'],
    total_fi_tracker_values['cpu'], total_fi_tracker_values['gpu'], total_fi_tracker_values['ram'], total_fi_tracker_values['duration']]
    
    df_results_for_seed = format_data_for_the_seed(columns_names, values)
    store_into_file('./results/Output_' + str(experiment_name), df_results_for_seed)
    _ = tracker.stop()


def main(): 
    dataset_name = "Backblaze"
    type_retraining_data = "SlidingWindow"
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
    TOTAL_NUMBER_SEEDS = 1
    random_seeds = list(np.arange(TOTAL_NUMBER_SEEDS))
    N_ITER_SEARCH = 100
    detections = ["StaticModel"] #"StaticModel" "PeriodicModel",

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
            pipeline_periodic_model(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
        if detection == "StaticModel":
             pipeline_static_model(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
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