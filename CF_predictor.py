import numpy as np
import IPython

def item_item_predict(training_matrix, test_set):
    # training_matrix.shape is num_user x num_movie
    num_user, num_movie = training_matrix.shape
    num_test = test_set.shape[1]

    # Find movie pearson correlation.
    training_matrix_dense = training_matrix.toarray()
    normalized_training_matrix = training_matrix_dense \
            - np.mean(training_matrix_dense, axis=0)
    col_norm = np.linalg.norm(normalized_training_matrix, axis=0)
    col_norm[col_norm == 0] = 1
    normalized_training_matrix = \
            normalized_training_matrix / col_norm
    similarity = normalized_training_matrix.T.dot(
                    normalized_training_matrix)

    prediction = np.empty(num_test)

    # Do prediction.
    default_rating = \
            (np.amin(training_matrix_dense) + \
            np.amax(training_matrix_dense)) / 2
    print "Setting default rating to", default_rating

    for i,index in enumerate(test_set.T):
        user = index[0]
        movie = index[1]
        weights = similarity[movie]
        rated_movies = (training_matrix_dense[user, :] != 0)
        # Normalize the weights so that their absolute
        # values sum to 1.
        normalization = np.sum(np.abs(weights[rated_movies]))
        if normalization == 0:
            print "Invalid normalizer in CF. Using default value."
            rating = default_rating
        else:
            rating = np.sum(training_matrix_dense[user] * weights)/ normalization
        prediction[i] = rating

    return prediction


