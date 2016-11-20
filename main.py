import sys
import parser_util
from scipy.io import mmread, mmwrite
from scipy.sparse import dok_matrix, lil_matrix
import baseline_predictor
import evaluator_util

def main(argv):
    FLAGS_parse_data = False
    data_dir = '../data/'
    rating_filename = 'ratings.csv'
    rating_matrixname = 'ratings.mtx'
    test_percentage = 0.1
    train_usage = 1.0
    method = 'baseline'
    evaluate_error_metric = 'RMSE'

    # Load data matrix.
    if FLAGS_parse_data:
        print "Data matrix does not exist. Parsing from raw data."
        rating_matrix = \
                parser_util.parse_user_rating_file( \
                data_dir+rating_filename)
    else:
        print "Loading data matrix from file."
        raw_matrix = mmread(data_dir+rating_matrixname)
        rating_matrix = raw_matrix.tolil()
        print "Num of users:", rating_matrix.shape[0], \
                ", Num of movies:", rating_matrix.shape[1]

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
