import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse
import IPython

def predict(training_matrix, test_set):
    # training_matrix.shape is num_user x num_movie
    num_user, num_movie = training_matrix.shape

    average_movie_rating = np.empty(num_movie, dtype=float)
    (x,y,z) = scipy.sparse.find(training_matrix)
    countings = np.bincount(y)
    sums = np.bincount(y, weights=z)
    average_movie_rating = sums/countings

    test_movies = test_set[1]
    prediction = average_movie_rating[test_movies]

    # Set those division by zero things to default rating.
    
    A = training_matrix[training_matrix.nonzero()].tocsr()
    default_rating = (A.max() + A.min())/2
    prediction[np.isnan(prediction)] = default_rating

    return prediction
