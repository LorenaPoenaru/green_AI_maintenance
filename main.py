import numpy as np
from tqdm import tqdm
from helpers import *
import sliding_window
import full_history

BACKBLAZE = "Backblaze"
GOOGLE = "Google"
ALIBABA = "Alibaba"



def main():
    dataset_name = ALIBABA
    if dataset_name == BACKBLAZE:
        DATASET_PATH_DISK = "./disk_2015_complete.csv"
        feature_list, label_list = features_labels_preprocessing(DATASET_PATH_DISK, 'b')
        features_failure = ['smart_1_raw', 'smart_4_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_12_raw', 'smart_187_raw', 'smart_193_raw', 'smart_194_raw', 'smart_197_raw', 'smart_199_raw', 
                         'smart_4_raw_diff', 'smart_5_raw_diff', 'smart_9_raw_diff', 'smart_12_raw_diff', 'smart_187_raw_diff', 'smart_193_raw_diff', 'smart_197_raw_diff', 'smart_199_raw_diff']
    elif dataset_name == GOOGLE:
        DATASET_PATH_DISK = "./google_job_failure.csv"
        feature_list, label_list = features_labels_preprocessing(DATASET_PATH_DISK, 'g')
        features_failure = ['User ID', 'Job Name', 'Scheduling Class',
                   'Num Tasks', 'Priority', 'Diff Machine', 'CPU Requested', 'Mem Requested', 'Disk Requested',
                   'Avg CPU', 'Avg Mem', 'Avg Disk', 'Std CPU', 'Std Mem', 'Std Disk']
    elif dataset_name == ALIBABA:
        DATASET_PATH_DISK = "./alibaba_job_data.csv"
        feature_list, label_list = features_labels_preprocessing(DATASET_PATH_DISK, 'a')
        features_failure = ['user', 'task_name', 'inst_num', 'plan_cpu', 'plan_mem', 'plan_gpu', 'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem']

    print(DATASET_PATH_DISK)
    num_chunks = len(feature_list)
    true_testing_labels = np.hstack(label_list[num_chunks//2:])
    initial_training_batches_list = list(range(0, num_chunks//2))
        
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

    configurations =  [("FullHistory", "Periodic"), ("FullHistory", "KS-ALL"), ("FullHistory", "KS-PCA"), ("FullHistory", "KS-FI"), ("SlidingWindow","Static"), ("SlidingWindow","Periodic"), ("SlidingWindow","KS-ALL"), ("SlidingWindow","KS-PCA"),("SlidingWindow","KS-FI")]
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
        if type_retraining_data == "SlidingWindow":
            if detection == "Periodic":
                sliding_window.pipeline_periodic_model(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
            if detection == "Static":
                sliding_window.pipeline_static_model(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
            if detection == "KS-ALL":
                sliding_window.pipeline_ks_all(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
            if detection == "KS-PCA":
                sliding_window.pipeline_ks_pca(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
            if detection == "KS-FI":
                sliding_window.pipeline_ks_fi(features_failure, dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
        elif type_retraining_data == "FullHistory":
            if detection == "Periodic":
                full_history.pipeline_periodic_model(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
            if detection == "KS-ALL":
                full_history.pipeline_ks_all(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
            if detection == "KS-PCA":
                full_history.pipeline_ks_pca(dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
            if detection == "KS-FI":
                full_history.pipeline_ks_fi(features_failure, dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
        counter[configuration] -= 1
    print("End of Experimentation")


if __name__ == "__main__":
    main()