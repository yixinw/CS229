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

def main(argv):
    FLAGS_parse_data = False
    FLAGS_toy = False
    # Set shrink_size=None if you don't want to shrink data set.
    shrink_size = [1000, 2000]
    data_dir = '../data/'
    rating_filename = 'ratings.csv'
    rating_matrixname = 'ratings_1000_2000.mtx'
    test_percentage = 0.1
    train_usage = 1.0
    method = 'CF'
    evaluate_error_metric = 'RMSE'

    # Load data matrix.
    if FLAGS_toy:
        print "Using toy example."
        rating_matrix = \
                parser_util.get_toy_dataset(
                        num_row=1000, num_col=2000,
                        rank=3, fill_rate=0.02)
    else:
        if FLAGS_parse_data:
            print "Data matrix does \
                    not exist. Parsing from raw data."
            rating_matrix = \
                    parser_util.parse_user_rating_file( \
                    data_dir+rating_filename, shrink_size)
            mmwrite(data_dir+rating_matrixname, rating_matrix)
        else:
            print "Loading data matrix from file."
            rating_matrix = parser_util.read_from_matrix(
                    data_dir+rating_matrixname, shrink_size)
            print "Num of users:", rating_matrix.shape[0], \
                    ", Num of movies:", rating_matrix.shape[1]
            print "Fill rate:", rating_matrix.nonzero()[0].shape[0] \
                    / float(rating_matrix.shape[0]*rating_matrix.shape[1])

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
        prediction = MGD_predictor.MGD1_predictor(training_matrix, test_set, r=3, trunc_flag=1)
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

if __name__ == '__main__':
    main(sys.argv)
