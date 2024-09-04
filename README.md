# green_AI_maintenance
I created this repository to work together on the sustainable model retraining project.

# Description:
We use in total 3 datasets, one is used to build disk failure prediction models (Backblaze), and the other two are used to build job failure prediction models (Google and Alibaba). We use 2 types of retraining from the data perspective, namely **sliding-window** in which we discard old data, and **full-history** in which we constantly enrich the training set with new data. We use 2 types of retraining from the retraining frequency perspective, namely **blind retraining**, in which the model is retrained periodically without checking whether there is drift (in the script this is called _Periodic Model_), and **informed retraining** in which the model is retrained only when the drift is detected. For informed retraining, we experiment with 3 drift detectors:  

- _KS on all features_ in which the distribution is estimated from all features used for training and testing, and then the KS statistical test is applied to the estimated distributions;
- _KS on PCA features_ in which PCA is initially applied on the features to reduce their dimensionality, then the distribution is estimated from the PCA features used for training and testing, and then the KS statistical test is applied to the estimated distributions;
- _KS on Important Features (FI)_ in which we only select the top most important features according to the model, then the distribution is estimated from the most important features used for training and testing, and then the KS statistical test is applied to the estimated distributions;

We also experiment with the situation in which we train once on the initial data and NEVER retrain afterward (in the script this is called _Static Model_)

# Results & Time Measuring
What do I currently measure & store in the results data frame:

1. Training Time - hyperparameter tunning time + time to fit the best configuration on the data
2. Hyperparameter Tunning time - the time to find the right hyperparameters
3. ROC_AUC + predictions + partial ROC_AUC per batch - to measure the performance of the models
4. Drift Detection Time - the total drift detection time is measured including all the steps required to detect drift (Drift Detection Total Time). According to each drift detector, different steps are measured:

- _KS on all features_ - we measure the time to estimate the distribution on all features (Distribution Extraction Time) + the time to apply the KS statistical test (Statistical Test Time)
- _KS on PCA features_ - we measure the time to apply PCA on the features (PCA Computing Time) + the time to estimate the distribution on all features (Distribution Extraction Time) + the time to apply the KS statistical test (Statistical Test Time)
- _KS on FI features_ - we measure the time to extract the most important features according to the model feature importance ranking (FI Extraction Time) + the time to estimate the distribution on all features (Distribution Extraction Time) + the time to apply the KS statistical test (Statistical Test Time)

Results are stored in different CSV files from the folder 'results'. We experiment with 30 different random seeds for results consistency.

# Requirements
pandas == 2.0.2

numpy == 1.25.1

scikit-learn == 1.2.2

scipy == 1.10.1

seaborn == 0.12.2

tqdm == 4.65.0

matplotlib == 3.7.2

# Running 

`docker build -t ai_maintenance .`

`docker run --rm -v <path to the project's folder>/results:/usr/src/app/results ai_maintenance main.py`
