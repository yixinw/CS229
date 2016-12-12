
# coding: utf-8

# In[ ]:

import numpy as np
import scipy.sparse as ss
import Manifold_descent
import IPython


# In[ ]:

def MGD1_predictor(training_matrix, test_set, r=5, trunc_flag=0):
    if trunc_flag:
        num_user, num_movie = training_matrix.shape

        mean_train = training_matrix[training_matrix.nonzero()].mean()
        temp = ss.lil_matrix(training_matrix - np.ones((num_user,num_movie))*mean_train)
        training_matrix = temp.multiply(training_matrix!=0)

        (x,y,z) = ss.find(training_matrix)
        countings_y = np.bincount(y)
        sums_y = np.bincount(y, weights=z)
        average_movie_rating_y = np.asmatrix(sums_y/countings_y)
        average_movie_rating_y[np.isnan(average_movie_rating_y)] = 0
        countings_x = np.bincount(x)
        sums_x = np.bincount(x, weights=z)
        average_movie_rating_x = np.asmatrix(sums_x/countings_x)
        average_movie_rating_x[np.isnan(average_movie_rating_x)] = 0

        temp = ss.lil_matrix(training_matrix - np.ones((num_user,1))*average_movie_rating_y-average_movie_rating_x.T * np.ones((1,num_movie)) )
        training_matrix = temp.multiply(training_matrix!=0)

    completed_mat = Manifold_descent.manifold_descent(training_matrix,r,30)
    result = np.asarray(completed_mat[tuple(test_set)])
    if trunc_flag:
        prediction_y = average_movie_rating_y[0,test_set[1]]
        prediction_x = average_movie_rating_x[0,test_set[0]]
        result += np.asarray(prediction_y + prediction_x + mean_train)
        result[result > 5] = 2.5
        result[result < 0.5] = 2.5
    return result


# In[ ]:

def MGD2_predictor(training_matrix, test_set, r=5, trunc_flag=0):
    if trunc_flag:
        num_user, num_movie = training_matrix.shape

        mean_train = training_matrix[training_matrix.nonzero()].mean()
        temp = ss.lil_matrix(training_matrix - np.ones((num_user,num_movie))*mean_train)
        training_matrix = temp.multiply(training_matrix!=0)

        (x,y,z) = ss.find(training_matrix)
        countings_y = np.bincount(y)
        sums_y = np.bincount(y, weights=z)
        average_movie_rating_y = np.asmatrix(sums_y/countings_y)
        average_movie_rating_y[np.isnan(average_movie_rating_y)] = 0
        countings_x = np.bincount(x)
        sums_x = np.bincount(x, weights=z)
        average_movie_rating_x = np.asmatrix(sums_x/countings_x)
        average_movie_rating_x[np.isnan(average_movie_rating_x)] = 0

        temp = ss.lil_matrix(training_matrix - np.ones((num_user,1))*average_movie_rating_y-average_movie_rating_x.T * np.ones((1,num_movie)) )
        training_matrix = temp.multiply(training_matrix!=0)

    completed_mat = Manifold_descent.manifold_descent_simplegrad(training_matrix,r,50)
    result = np.asarray(completed_mat[tuple(test_set)])
    if trunc_flag:
        prediction_y = average_movie_rating_y[0,test_set[1]]
        prediction_x = average_movie_rating_x[0,test_set[0]]
        result += np.asarray(prediction_y + prediction_x + mean_train)
        result[result > 5] = 2.5
        result[result < 0.5] = 2.5
    return result

