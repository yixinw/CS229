import numpy as np

def evaluate(test_set, prediction,ground_truth, error_metric):
    # Get an helper function according to error metric.
    if error_metric == 'RMSE':
        return get_RMSE(test_set, prediction, ground_truth)
    else:
        print "Invalid error metric for evaluation."
        return 0.0


def get_RMSE(test_set, prediction, ground_truth):
    actual = ground_truth[tuple(test_set)].toarray()
    square_error = np.nanmean((actual - prediction) ** 2)
    return np.sqrt(square_error)
