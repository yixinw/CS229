
# coding: utf-8

# In[ ]:

import numpy as np
import scipy.sparse as ss
import Manifold_descent
import IPython


# In[ ]:

def MGD1_predictor(training_matrix, test_set, r=5, trunc_flag=0):
    completed_mat = Manifold_descent.manifold_descent(training_matrix,r)
    result = np.asarray(completed_mat[tuple(test_set)])
    if trunc_flag:
        result[result > 5] = 5
        result[result < 0.5] = 0.5
    return result


# In[ ]:

def MGD2_predictor(training_matrix, test_set, r=5, trunc_flag=0):
    completed_mat = Manifold_descent.manifold_descent_simplegrad(training_matrix,r)
    result = np.asarray(completed_mat[tuple(test_set)])
    if trunc_flag:
        result[result > 5] = 5
        result[result < 0.5] = 0.5
    return result

