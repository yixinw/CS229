# %load main.py
import sys
import parser_util
from scipy.io import mmread, mmwrite
from scipy.sparse import dok_matrix, lil_matrix
import baseline_predictor
import evaluator_util
import MGD_predictor
import CF_predictor
import IPython
import numpy as np
import scipy.io as sio

def run(fill_rate, noise, method):
    test_percentage = 0.1
    train_usage = 1.0
    method = method
    evaluate_error_metric = 'RMSE'

    # Load data matrix.
    print "Using toy example."
    rating_matrix = \
            parser_util.get_toy_dataset(
                    num_row=600, num_col=1000,
                    rank=5, fill_rate=fill_rate,
                    bias_flag=1, noise_level=noise)

    # Split into training and test set.
    print "Setting up training and test set."
    test_set = parser_util.get_test_set(
            test_percentage, rating_matrix)
    training_matrix = parser_util.get_training_matrix(
            train_usage, test_set, rating_matrix)
    print "Test set proportion:", test_percentage
    print "Training s_usage:", train_usage

    # Predict.
    print "Doing prediction."
    if method == 'baseline':
        prediction = baseline_predictor.predict(
                training_matrix=training_matrix,
                test_set=test_set)
    elif method == 'MGD1':
        prediction = MGD_predictor.MGD1_predictor(training_matrix, test_set, r=7, trunc_flag=0)
    elif method == 'MGD2':
        prediction = MGD_predictor.MGD2_predictor(training_matrix, test_set, trunc_flag=1)
    elif method == 'CF':
        prediction = CF_predictor.item_item_predict(
                training_matrix=training_matrix,
                test_set=test_set)
    else:
        print "Invalid prediction method."

    # Evaluate prediction.
    print "Evaluating prediction result."
    RMSE = evaluator_util.evaluate(
            test_set=test_set,
            prediction=prediction,
            ground_truth=rating_matrix,
            error_metric=evaluate_error_metric)
    print RMSE
    return RMSE

# Run three cases of toy example to control noise
# and fill_rate.

# Case 1: Noise free, fill rate 0.05, 0.1, ..., 0.3.
data_dir = '../data/'
method = 'MGD1'
RMSE_list = np.zeros(6)
noise = 0
fill_rate_list = np.linspace(0.05, 0.3, num=6)
for i,fill_rate in enumerate(fill_rate_list):
    RMSE = run(fill_rate=fill_rate, noise=noise, method=method)
    RMSE_list[i] = RMSE
print "Finished case 1. RMSE is"
print RMSE_list
sio.savemat(data_dir+method+'_case1.mat', {method+'_case1':RMSE_list})

# Case 2: Constant noise, fill rate 0.05, 0.1, ..., 0.3.
RMSE_list = np.zeros(6)
noise = 0.1
fill_rate_list = np.linspace(0.05, 0.3, num=6)
for i,fill_rate in enumerate(fill_rate_list):
    RMSE = run(fill_rate=fill_rate, noise=noise, method=method)
    RMSE_list[i] = RMSE
print "Finished case 2. RMSE is"
print RMSE_list
sio.savemat(data_dir+method+'_case2.mat', {method+'_case2':RMSE_list})

# Case 3: Noise 0.001, 0.005,
RMSE_list = np.zeros(7)
fill_rate = 0.3
noise_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
for i,noise in enumerate(noise_list):
    RMSE = run(fill_rate=fill_rate, noise=noise, method=method)
    RMSE_list[i] = RMSE
print "Finished case 3. RMSE is"
print RMSE_list
sio.savemat(data_dir+method+'_case3.mat', {method+'_case3':RMSE_list})



