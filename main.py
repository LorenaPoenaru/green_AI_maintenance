import numpy as np
from helpers import *
from sliding_window import *

def main(): 
    dataset_name = "Backblaze"
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

    configurations = [("SlidingWindow","Static"), ("SlidingWindow","Periodic"), 
    ("SlidingWindow","KS-ALL"), ("SlidingWindow","KS-PCA"),("SlidingWindow","KS-FI")]

    counter = {}
    for configuration in configurations:
        counter[configuration] = TOTAL_NUMBER_SEEDS


    executions = configurations * TOTAL_NUMBER_SEEDS
    random.shuffle(executions)
    print(executions)

    for configuration in executions:
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
                sliding_window.pipeline_ks_fi(features_disk_failure, dataset_name, type_retraining_data, detection, random_seed,feature_list, label_list, num_chunks, param_dist_rf, N_ITER_SEARCH, true_testing_labels, initial_training_batches_list)
        counter[configuration] -= 1
    print("End of Experimentation")


if __name__ == "__main__":
    main()