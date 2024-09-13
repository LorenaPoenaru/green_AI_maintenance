
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


BACKBLAZE = "Backblaze"
GOOGLE = "Google"
ALIBABA = "Alibaba"




def pipeline_periodic_model(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list):
    experiment_name = str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(random_seed)
    
    total_time_training = 0
    predictions_test_fh = []
    partial_roc_auc_fh = []
    total_train_fh = 0 
    total_hyperparam_fh = 0
    total_test_fh = 0
    
    ### Tracker tasks variables 
    tracker = EmissionsTracker(project_name="AI_maintenance_" + str(experiment_name))
    total_hyperparam_tracker_values, total_fit_tracker_values, total_testing_tracker_values = initiate_tracker_variables()
    
    begin_total_fh = time.time()
    for batch in tqdm(range(num_chunks//2, num_chunks)):
        training_features_init = np.vstack(feature_list[0: batch])
        training_labels_init = np.hstack(label_list[0//2: batch])
        
    
        # check if it is the first batch
        if(batch==num_chunks//2):
            training_features = training_features_init
            training_labels = training_labels_init
        

        print("BATCH", batch)
        print("training_features BEFORE SCALING", training_features, len(training_features))
        # scaler for training data
        update_scaler = StandardScaler()
        training_features_model = update_scaler.fit_transform(training_features)
        training_labels_model = training_labels

        # obtain testing features and labels
        testing_features = feature_list[batch]
        testing_labels = label_list[batch]
        print("testing_features BEFORE SCALING", testing_features, len(testing_features)) 
        
        # scaling testing features
        testing_features_model = update_scaler.transform(testing_features)
        print("testing_features AFTER", testing_features, len(testing_features)) 
        print("training_features AFTER", training_features, len(training_features))
        #print("testing_labels",testing_labels, len(testing_labels))


        # Downscaling for data training
        if dataset_name != ALIBABA:
            training_features_processed, training_labels_processed = downsampling(training_features_model, training_labels_model, random_seed)
        else:
            training_features_processed = training_features_model
            training_labels_processed = training_labels_model
        # training model
        begin_train_fh = time.time()
        begin_hyperparam_tunning_update = time.time()
        update_model, total_hyperparam_tracker_values = hyperparameter_tuning_process(dataset_name, type_retraining_data, 
                                                                                          detection, random_seed, batch, param_dist_rf, 
                                                                                          N_ITER_SEARCH, training_features_processed, 
                                                                                          training_labels_processed,
                                                                                          total_hyperparam_tracker_values, tracker)
        end_hyperparam_tunning_update = time.time() - begin_hyperparam_tunning_update
        # Hyperparameter tunning energy collection end
        update_model, total_fit_tracker_values = best_model_fit(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, training_features_processed, training_labels_processed, total_fit_tracker_values, tracker)
        # Fitting the model energy collection end
        end_train_fh = time.time() - begin_train_fh
        total_hyperparam_fh = total_hyperparam_fh + end_hyperparam_tunning_update
        total_train_fh = total_train_fh + end_train_fh
        
        
        # evaluate model on testing data
        begin_test_fh = time.time()
        predictions_test_updated, total_testing_tracker_values = get_predictions(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, testing_features_model, total_testing_tracker_values, tracker)
        end_test_fh = time.time() - begin_test_fh
        total_test_fh = total_test_fh + end_test_fh


        partial_roc_auc_fh.append(roc_auc_score(testing_labels, predictions_test_updated))
        predictions_test_fh = np.concatenate([predictions_test_fh, predictions_test_updated])

        training_features = np.vstack(feature_list[0: batch+1])
        training_labels = np.hstack(label_list[0: batch+1])


    end_total_fh = time.time() - begin_total_fh
    
    
    columns_names =['Random_Seed', 'Model', 'Drifts', 'ROC_AUC_Batch', 'ROC_AUC_BATCH_MEAN', 'ROC_AUC_Total', 
    'Predictions', 'True_Testing_Labels', 'Train_Time', 'Hyperparam_Tunning_Time', 'Test_Time', 'Drifts_Detected', 'Label_Costs',
    'Energy_Consumed_Hyperparameter', 'Emissions_Hyperparameter', 'CPU_Energy_Hyperparameter', 'GPU_Energy_Hyperparameter', 'RAM_Energy_Hyperparameter', 'Duration_Tracker_Hyperparameter', 
    'Energy_Consumed_Fitting','Emissions_Fitting', 'CPU_Energy_Fitting', 'GPU_Energy_Fitting', 'RAM_Energy_Fitting', 'Duration_Tracker_Fitting', 
    'Energy_Consumed_Testing', 'Emissions_Testing', 'CPU_Energy_Testing', 'GPU_Energy_Testing', 'RAM_Energy_Testing', 'Duration_Tracker_Testing']
    values =  [random_seed, type_retraining_data+"_"+detection, str(int(num_chunks//2)) + '/' + str(int(num_chunks//2)), 
    partial_roc_auc_fh, np.mean(partial_roc_auc_fh), roc_auc_score(true_testing_labels, predictions_test_fh), 
    predictions_test_fh, true_testing_labels,  total_train_fh, total_hyperparam_fh, total_test_fh, np.ones(int(num_chunks//2), dtype=int), 
    len(true_testing_labels),
    total_hyperparam_tracker_values['energy_consumed'], total_hyperparam_tracker_values['emissions'], total_hyperparam_tracker_values['cpu'], total_hyperparam_tracker_values['gpu'], total_hyperparam_tracker_values['ram'], total_hyperparam_tracker_values['duration'], 
    total_fit_tracker_values['energy_consumed'], total_fit_tracker_values['emissions'], total_fit_tracker_values['cpu'], total_fit_tracker_values['gpu'], total_fit_tracker_values['ram'], total_fit_tracker_values['duration'], 
    total_testing_tracker_values['energy_consumed'], total_testing_tracker_values['emissions'], total_testing_tracker_values['cpu'], total_testing_tracker_values['gpu'], total_testing_tracker_values['ram'], total_testing_tracker_values['duration']
    ]
    df_results_for_seed = format_data_for_the_seed(columns_names, values)
    store_into_file('./results/Output_' + str(experiment_name) + ".csv", df_results_for_seed)
    _ = tracker.stop()




# # # Build Drift Detection based Model Update
# # ### KS on all features
def pipeline_ks_all(dataset_name, type_retraining_data, detection, random_seed, feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list):
    experiment_name = str(dataset_name) + "_" + str(type_retraining_data) + "_" + str(detection) + "_" + str(random_seed)
    print(experiment_name) 

    #print("feature_list", feature_list)
    #print("label_list", label_list)
    #for i in range(0, len(feature_list)):
    #    print(len(feature_list[i]))

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
    
    
    for batch in tqdm(range(num_chunks//2, num_chunks)):
        training_features_init = np.vstack(feature_list[0: batch])
        training_labels_init = np.hstack(label_list[0//2: batch])
        
        # init drift alert
        drift_alert = 0
    
        # check if it is the first batch
        if(batch==num_chunks//2):
            training_features = training_features_init
            training_labels = training_labels_init
            current_training_batches_list = initial_training_batches_list.copy()
            print('Initial Training Batches', current_training_batches_list)
        

        print("BATCH", batch)
        print("training_features BEFORE SCALING", training_features, len(training_features))
        # scaler for training data
        update_scaler = StandardScaler()
        training_features_model = update_scaler.fit_transform(training_features)
        training_labels_model = training_labels
        print("training_features AFTER", training_features_model, len(training_features_model))

        # Downscaling for data training
        if dataset_name != ALIBABA:
            training_features_processed, training_labels_processed = downsampling(training_features_model, training_labels_model, random_seed)
        else:
            training_features_processed = training_features_model
            training_labels_processed = training_labels_model

        # obtain testing features and labels
        testing_features = feature_list[batch]
        testing_labels = label_list[batch]
        print("testing_features BEFORE SCALING", testing_features, len(testing_features)) 
        
        # scaling testing features
        testing_features_model = update_scaler.transform(testing_features)
        testing_labels_model = testing_labels
        print("testing_features AFTER SCALING", testing_features_model, len(testing_features_model)) 

        # training model
        begin_train_fh_ks_all = time.time()
        if(batch==num_chunks//2 or need_to_retrain == 1):
            begin_train_fh_ks_all = time.time()
            begin_hyperparam_tunning_update = time.time()
            update_model, total_hyperparam_tracker_values = hyperparameter_tuning_process(dataset_name, type_retraining_data, 
                                                                                          detection, random_seed, batch, param_dist_rf, 
                                                                                          N_ITER_SEARCH, training_features_processed, 
                                                                                          training_labels_processed,
                                                                                          total_hyperparam_tracker_values, tracker)
            end_hyperparam_tunning_update = time.time() - begin_hyperparam_tunning_update
            total_hyperparam_fh_ks_all = total_hyperparam_fh_ks_all + end_hyperparam_tunning_update

            update_model, total_fit_tracker_values = best_model_fit(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, training_features_processed, training_labels_processed, total_fit_tracker_values, tracker)
            end_train_fh_ks_all = time.time() - begin_train_fh_ks_all
            total_train_fh_all = total_train_fh_all + end_train_fh_ks_all
        
        # evaluate model on testing data
        begin_test_time_ks_all = time.time()
        predictions_test_updated, total_testing_tracker_values = get_predictions(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, testing_features_model, total_testing_tracker_values, tracker)
        end_test_time_ks_all = time.time() - begin_test_time_ks_all
        total_test_time_ks_all = total_test_time_ks_all + end_test_time_ks_all

        partial_roc_auc_ks_all_model.append(roc_auc_score(testing_labels_model, predictions_test_updated)) 
        predictions_test_ks_all_model = np.concatenate([predictions_test_ks_all_model, predictions_test_updated])
        
        # Drift Detection
        need_to_retrain = 0        
        
        drift_time_start = time.time()
        drift_alert, distribution_extraction_time, ks_test_time, total_distribution_tracker_values, total_stats_tracker_values = ks_drift_detection(dataset_name, type_retraining_data, detection, random_seed, batch, training_features_processed, testing_features_model, total_distribution_tracker_values, total_stats_tracker_values, tracker)
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
                     'Energy_Consumed_Hyperparameter', 'Emissions_Hyperparameter', 'CPU_Energy_Hyperparameter', 'GPU_Energy_Hyperparameter', 'RAM_Energy_Hyperparameter', 'Duration_Tracker_Hyperparameter', 
                     'Energy_Consumed_Fitting','Emissions_Fitting', 'CPU_Energy_Fitting', 'GPU_Energy_Fitting', 'RAM_Energy_Fitting', 'Duration_Tracker_Fitting', 
                     'Energy_Consumed_Testing', 'Emissions_Testing', 'CPU_Energy_Testing', 'GPU_Energy_Testing', 'RAM_Energy_Testing', 'Duration_Tracker_Testing'
                     'Energy_Consumed_Distribution_Extraction', 'Emissions_Distribution_Extraction', 'CPU_Energy_Distribution_Extraction', 'GPU_Energy_Distribution_Extraction', 'RAM_Energy_Distribution_Extraction', 'Duration_Tracker_Distribution_Extraction',
                     'Energy_Consumed_Stats_Test', 'Emissions_Stats_Test', 'CPU_Energy_Stats_Test', 'GPU_Energy_Stats_Test', 'RAM_Energy_Stats_Test', 'Duration_Tracker_Stats_Test'
                     ]
    values = [random_seed, type_retraining_data+"_"+detection, str(no_necessary_retrainings)+'/'+str(len(detected_drifts)), partial_roc_auc_ks_all_model, 
              np.mean(partial_roc_auc_ks_all_model), roc_auc_score(true_testing_labels, predictions_test_ks_all_model), 
              predictions_test_ks_all_model, true_testing_labels, total_train_fh_all, total_hyperparam_fh_ks_all, 
              total_test_time_ks_all, detected_drifts, total_drift_detection_time, total_distribution_extraction_time, 
              total_stat_test_time, necessary_label_annotation_effort,
              total_hyperparam_tracker_values['energy_consumed'], total_hyperparam_tracker_values['emissions'], total_hyperparam_tracker_values['cpu'], total_hyperparam_tracker_values['gpu'], total_hyperparam_tracker_values['ram'], total_hyperparam_tracker_values['duration'], 
              total_fit_tracker_values['energy_consumed'], total_fit_tracker_values['emissions'], total_fit_tracker_values['cpu'], total_fit_tracker_values['gpu'], total_fit_tracker_values['ram'], total_fit_tracker_values['duration'], 
              total_testing_tracker_values['energy_consumed'], total_testing_tracker_values['emissions'], total_testing_tracker_values['cpu'], total_testing_tracker_values['gpu'], total_testing_tracker_values['ram'], total_testing_tracker_values['duration'],
              total_distribution_tracker_values['energy_consumed'], total_distribution_tracker_values['emissions'], total_distribution_tracker_values['cpu'], total_distribution_tracker_values['gpu'], total_distribution_tracker_values['ram'], total_distribution_tracker_values['duration'],
              total_stats_tracker_values['energy_consumed'], total_stats_tracker_values['emissions'], total_stats_tracker_values['cpu'], total_stats_tracker_values['gpu'], total_stats_tracker_values['ram'], total_stats_tracker_values['duration']]
    
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
    
    training_features, training_labels = initiate_training_features_and_labels_from_lists(feature_list, label_list, num_chunks)
    current_training_batches_list = initial_training_batches_list.copy()
    #need_to_retrain = 0
    for batch in tqdm(range(num_chunks//2, num_chunks)):
        training_features_init = np.vstack(feature_list[0: batch])
        training_labels_init = np.hstack(label_list[0//2: batch])
        
        # init drift alert
        drift_alert = 0
    
        # check if it is the first batch
        if(batch==num_chunks//2):
            training_features = training_features_init
            training_labels = training_labels_init
            current_training_batches_list = initial_training_batches_list.copy()
            print('Initial Training Batches', current_training_batches_list)
        

        print("BATCH", batch)
        print("training_features BEFORE SCALING", training_features, len(training_features))
        # scaler for training data
        update_scaler = StandardScaler()
        training_features_model = update_scaler.fit_transform(training_features)
        training_labels_model = training_labels

        # obtain testing features and labels
        testing_features = feature_list[batch]
        testing_labels = label_list[batch]
        print("testing_features BEFORE SCALING", testing_features, len(testing_features)) 
        
        # scaling testing features
        testing_features_model = update_scaler.transform(testing_features)
        print("testing_features AFTER", testing_features_model, len(testing_features_model)) 
        print("training_features AFTER", training_features_model, len(training_features_model))
        #print("testing_labels",testing_labels, len(testing_labels))
        # Downscaling for data training
        if dataset_name != ALIBABA:
            training_features_processed, training_labels_processed = downsampling(training_features_model, training_labels_model, random_seed)
        else:
            training_features_processed = training_features_model
            training_labels_processed = training_labels_model
    
        begin_train_fh_ks_pca = time.time()
        if(batch==num_chunks//2 or need_to_retrain == 1):
            begin_train_fh_ks_pca = time.time()
            begin_hyperparam_tunning_update = time.time()
            update_model, total_hyperparam_tracker_values = hyperparameter_tuning_process(dataset_name, type_retraining_data, 
                                                                                          detection, random_seed, batch, param_dist_rf, 
                                                                                          N_ITER_SEARCH, training_features_processed, 
                                                                                          training_labels_processed,
                                                                                          total_hyperparam_tracker_values, tracker)
            end_hyperparam_tunning_update = time.time() - begin_hyperparam_tunning_update
            
            total_hyperparam_fh_ks_pca = total_hyperparam_fh_ks_pca + end_hyperparam_tunning_update
            
            update_model, total_fit_tracker_values = best_model_fit(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, training_features_processed, training_labels_processed, total_fit_tracker_values, tracker)
            end_train_fh_ks_pca = time.time() - begin_train_fh_ks_pca
            total_train_fh_pca = total_train_fh_pca + end_train_fh_ks_pca
        
        
        # evaluate model on testing data & measure testing time
        begin_test_time_ks_pca = time.time()
        predictions_test_updated, total_testing_tracker_values = get_predictions(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, testing_features_model, total_testing_tracker_values, tracker)
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
        df_train_features_sorted_pca, df_test_features_sorted_pca, total_pca_tracker_values = run_pca(dataset_name, type_retraining_data, detection, random_seed, batch, training_features_processed, testing_features_model, total_pca_tracker_values, tracker)
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
    'Energy_Consumed_Hyperparameter', 'Emissions_Hyperparameter', 'CPU_Energy_Hyperparameter', 'GPU_Energy_Hyperparameter', 'RAM_Energy_Hyperparameter', 'Duration_Tracker_Hyperparameter', 
    'Energy_Consumed_Fitting','Emissions_Fitting', 'CPU_Energy_Fitting', 'GPU_Energy_Fitting', 'RAM_Energy_Fitting', 'Duration_Tracker_Fitting', 
    'Energy_Consumed_Testing', 'Emissions_Testing', 'CPU_Energy_Testing', 'GPU_Energy_Testing', 'RAM_Energy_Testing', 'Duration_Tracker_Testing',
    'Energy_Consumed_Distribution_Extraction', 'Emissions_Distribution_Extraction', 'CPU_Energy_Distribution_Extraction', 'GPU_Energy_Distribution_Extraction', 'RAM_Energy_Distribution_Extraction', 'Duration_Tracker_Distribution_Extraction',
    'Energy_Consumed_Stats_Test', 'Emissions_Stats_Test', 'CPU_Energy_Stats_Test', 'GPU_Energy_Stats_Test', 'RAM_Energy_Stats_Test', 'Duration_Tracker_Stats_Test',
    'Energy_Consumed_PCA', 'Emissions_PCA', 'CPU_Energy_PCA', 'GPU_Energy_PCA', 'RAM_Energy_PCA', 'Duration_Tracker_PCA'
    ]
    values=[random_seed, type_retraining_data+"_"+detection, str(no_necessary_retrainings)+'/'+str(len(detected_drifts)), 
    partial_roc_auc_ks_pca_model, np.mean(partial_roc_auc_ks_pca_model), 
    roc_auc_score(true_testing_labels, predictions_test_ks_pca_model), 
    predictions_test_ks_pca_model, true_testing_labels, total_train_fh_pca, 
    total_hyperparam_fh_ks_pca, total_test_time_ks_pca, detected_drifts, 
    total_drift_detection_time, total_pca_time, total_distribution_extraction_time, 
    total_stat_test_time, necessary_label_annotation_effort,
    total_hyperparam_tracker_values['energy_consumed'], total_hyperparam_tracker_values['emissions'], total_hyperparam_tracker_values['cpu'], total_hyperparam_tracker_values['gpu'], total_hyperparam_tracker_values['ram'], total_hyperparam_tracker_values['duration'], 
    total_fit_tracker_values['energy_consumed'], total_fit_tracker_values['emissions'], total_fit_tracker_values['cpu'], total_fit_tracker_values['gpu'], total_fit_tracker_values['ram'], total_fit_tracker_values['duration'], 
    total_testing_tracker_values['energy_consumed'], total_testing_tracker_values['emissions'], total_testing_tracker_values['cpu'], total_testing_tracker_values['gpu'], total_testing_tracker_values['ram'], total_testing_tracker_values['duration'],
    total_distribution_tracker_values['energy_consumed'], total_distribution_tracker_values['emissions'], total_distribution_tracker_values['cpu'], total_distribution_tracker_values['gpu'], total_distribution_tracker_values['ram'], total_distribution_tracker_values['duration'],
    total_stats_tracker_values['energy_consumed'], total_stats_tracker_values['emissions'], total_stats_tracker_values['cpu'], total_stats_tracker_values['gpu'], total_stats_tracker_values['ram'], total_stats_tracker_values['duration'],
    total_pca_tracker_values['energy_consumed'], total_pca_tracker_values['emissions'], total_pca_tracker_values['cpu'], total_pca_tracker_values['gpu'], total_pca_tracker_values['ram'], total_pca_tracker_values['duration']]

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
    
    training_features, training_labels = initiate_training_features_and_labels_from_lists(feature_list, label_list, num_chunks)
    current_training_batches_list = initial_training_batches_list.copy()
    #need_to_retrain = 0
    for batch in tqdm(range(num_chunks//2, num_chunks)):
        training_features_init = np.vstack(feature_list[0: batch])
        training_labels_init = np.hstack(label_list[0//2: batch])
        
        # init drift alert
        drift_alert = 0
    
        if(batch==num_chunks//2):
            training_features = training_features_init
            training_labels = training_labels_init
            current_training_batches_list = initial_training_batches_list.copy()
            print('Initial Training Batches', current_training_batches_list)
        

        print("BATCH", batch)
        print("training_features BEFORE SCALING", training_features, len(training_features))
        # scaler for training data
        update_scaler = StandardScaler()
        training_features_model = update_scaler.fit_transform(training_features)
        training_labels_model = training_labels

        # obtain testing features and labels
        testing_features = feature_list[batch]
        testing_labels = label_list[batch]
        print("testing_features BEFORE SCALING", testing_features, len(testing_features)) 
        
        # scaling testing features
        testing_features_model = update_scaler.transform(testing_features)
        print("testing_features AFTER", testing_features_model, len(testing_features_model)) 
        print("training_features AFTER", training_features_model, len(training_features_model))
        #print("testing_labels",testing_labels, len(testing_labels))
        # Downscaling for data training
               # Downscaling for data training
        if dataset_name != ALIBABA:
            training_features_processed, training_labels_processed = downsampling(training_features_model, training_labels_model, random_seed)
        else:
            training_features_processed = training_features_model
            training_labels_processed = training_labels_model

        # training model
        begin_train_fh_ks_fi = time.time()
        if(batch==num_chunks//2 or need_to_retrain == 1):
            begin_train_fh_ks_fi = time.time()
            begin_hyperparam_tunning_update = time.time()
            update_model, total_hyperparam_tracker_values = hyperparameter_tuning_process(dataset_name, type_retraining_data, 
                                                                                          detection, random_seed, batch, param_dist_rf, 
                                                                                          N_ITER_SEARCH, training_features_processed, 
                                                                                          training_labels_processed,
                                                                                          total_hyperparam_tracker_values, tracker)
            end_hyperparam_tunning_update = time.time() - begin_hyperparam_tunning_update
            total_hyperparam_fh_ks_fi = total_hyperparam_fh_ks_fi + end_hyperparam_tunning_update
            
            update_model, total_fit_tracker_values = best_model_fit(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, training_features_processed, training_labels_processed, total_fit_tracker_values, tracker)
            end_train_fh_ks_fi = time.time() - begin_train_fh_ks_fi
            total_train_fh_fi = total_train_fh_fi + end_train_fh_ks_fi
        
        
#         # evaluate model on testing data
        
        begin_test_time_ks_fi = time.time()
        predictions_test_updated, total_testing_tracker_values = get_predictions(dataset_name, type_retraining_data, detection, random_seed, batch, update_model, testing_features_model, total_testing_tracker_values, tracker)
        end_test_time_ks_fi = time.time() - begin_test_time_ks_fi
        total_test_time_ks_fi = total_test_time_ks_fi + end_test_time_ks_fi
        
        partial_roc_auc_ks_fi_model.append(roc_auc_score(testing_labels, predictions_test_updated))
        predictions_test_ks_fi_model = np.concatenate([predictions_test_ks_fi_model, predictions_test_updated])
        

        # Drift Detection
        need_to_retrain = 0
        drift_time_start = time.time()
        # Extract Most Important Features
        feature_importance_extraction_start = time.time()  
        important_features, training_important_features_model, testing_important_features_model, total_fi_tracker_values = get_fi(dataset_name, type_retraining_data, detection, random_seed, batch, tracker, total_fi_tracker_values, update_model, training_features_processed, testing_features_model, features_disk_failure)
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
    
    columns_names =['Random_Seed', 'Model', 'Drifts_Overall',  'ROC_AUC_Batch', 'ROC_AUC_BATCH_MEAN', 
    'ROC_AUC_Total', 'Predictions', 'True_Testing_Labels', 'Train_Time', 'Hyperparam_Tunning_Time', 
    'Test_Time', 'Drifts_Detected', 'Drift_Detection_Total_Time', 'FI_Extraction_Time', 
    'Distribution_Extraction_Time', 'Statistical_Test_Time', 'Label_Costs',
    'Energy_Consumed_Hyperparameter', 'Emissions_Hyperparameter', 'CPU_Energy_Hyperparameter', 'GPU_Energy_Hyperparameter', 'RAM_Energy_Hyperparameter', 'Duration_Tracker_Hyperparameter', 
    'Energy_Consumed_Fitting','Emissions_Fitting', 'CPU_Energy_Fitting', 'GPU_Energy_Fitting', 'RAM_Energy_Fitting', 'Duration_Tracker_Fitting', 
    'Energy_Consumed_Testing', 'Emissions_Testing', 'CPU_Energy_Testing', 'GPU_Energy_Testing', 'RAM_Energy_Testing', 'Duration_Tracker_Testing',
    'Energy_Consumed_Distribution_Extraction', 'Emissions_Distribution_Extraction', 'CPU_Energy_Distribution_Extraction', 'GPU_Energy_Distribution_Extraction', 'RAM_Energy_Distribution_Extraction', 'Duration_Tracker_Distribution_Extraction',
    'Energy_Consumed_Stats_Test', 'Emissions_Stats_Test', 'CPU_Energy_Stats_Test', 'GPU_Energy_Stats_Test', 'RAM_Energy_Stats_Test', 'Duration_Tracker_Stats_Test',
    'Energy_Consumed_FI', 'Emissions_FI', 'CPU_Energy_FI', 'GPU_Energy_FI', 'RAM_Energy_FI', 'Duration_Tracker_FI']
    values= [random_seed, type_retraining_data+"_"+detection, str(no_necessary_retrainings)+'/'+str(len(detected_drifts)), 
    partial_roc_auc_ks_fi_model, np.mean(partial_roc_auc_ks_fi_model), roc_auc_score(true_testing_labels, 
    predictions_test_ks_fi_model), predictions_test_ks_fi_model, true_testing_labels, total_train_fh_fi, 
    total_hyperparam_fh_ks_fi, total_test_time_ks_fi, detected_drifts, total_drift_detection_time, total_feature_importance_extraction_time, 
    total_distribution_extraction_time, total_stat_test_time, necessary_label_annotation_effort,
    total_hyperparam_tracker_values['energy_consumed'], total_hyperparam_tracker_values['emissions'], total_hyperparam_tracker_values['cpu'], total_hyperparam_tracker_values['gpu'], total_hyperparam_tracker_values['ram'], total_hyperparam_tracker_values['duration'], 
    total_fit_tracker_values['energy_consumed'], total_fit_tracker_values['emissions'], total_fit_tracker_values['cpu'], total_fit_tracker_values['gpu'], total_fit_tracker_values['ram'], total_fit_tracker_values['duration'], 
    total_testing_tracker_values['energy_consumed'], total_testing_tracker_values['emissions'], total_testing_tracker_values['cpu'], total_testing_tracker_values['gpu'], total_testing_tracker_values['ram'], total_testing_tracker_values['duration'],
    total_distribution_tracker_values['energy_consumed'], total_distribution_tracker_values['emissions'], total_distribution_tracker_values['cpu'], total_distribution_tracker_values['gpu'], total_distribution_tracker_values['ram'], total_distribution_tracker_values['duration'],
    total_stats_tracker_values['energy_consumed'], total_stats_tracker_values['emissions'], total_stats_tracker_values['cpu'], total_stats_tracker_values['gpu'], total_stats_tracker_values['ram'], total_stats_tracker_values['duration'],
    total_fi_tracker_values['energy_consumed'], total_fi_tracker_values['emissions'], total_fi_tracker_values['cpu'], total_fi_tracker_values['gpu'], total_fi_tracker_values['ram'], total_fi_tracker_values['duration']]
    
    df_results_for_seed = format_data_for_the_seed(columns_names, values)
    store_into_file('./results/Output_' + str(experiment_name) + ".csv", df_results_for_seed)
    _ = tracker.stop()
    
