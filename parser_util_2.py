'''
This util file helps parsing rating data file into a huge sparse matrix.
Each row corresponds to a user, each column corresponds to a movie.

Sample usage:
data_dir = '../data/'
filename = 'ratings.csv'
rating_matrix = parse_user_rating_file(data_dir+filename)

out_filename = 'ratings.mtx'
print "######## Saving to disk ########"
mmwrite(data_dir + out_filename, rating_matrix)
# To read this matrix, use the following command.
# A = mmread('ratings.mtx')
'''

import csv
from scipy.sparse import lil_matrix, dok_matrix
import IPython
import numpy as np
from scipy.io import mmread, mmwrite
import random
import IPython

def parse_user_rating_file(filename, shrink_size):
    # First pass, count the number of user and items (movies).
    print "####### Start parsing #######"
    user_dict = {}
    movie_dict = {}
    with open(filename, 'r') as f:
        f.readline()
        file_iter = csv.reader(f, delimiter=',')
        iter_counter = 0
        for row in file_iter:
            iter_counter += 1
            if iter_counter % 1000000 == 0:
                print iter_counter
            user_id = row[0]
            movie_id = row[1]
            user_dict[user_id] = None
            movie_dict[movie_id] = None
    num_user = len(user_dict)
    num_movie = len(movie_dict)
    print "Num of user is:", num_user, ", Num of movie is:", num_movie

    rating_matrix = lil_matrix((num_user, num_movie)) # size: num_user x num_movie

    # Second pass, populate the rating matrix.
    print "####### Start populating rating matrix #######"
    user_dict = {}
    movie_dict = {}
    user_counter = 0
    movie_counter = 0
    with open(filename, 'r') as f:
        f.readline()
        file_iter = csv.reader(f, delimiter=',')
        iter_counter = 0
        for row in file_iter:
            iter_counter += 1
            if iter_counter % 1000000 == 0:
                print iter_counter
            user_id = row[0]
            movie_id = row[1]
            rating = float(row[2])
            if user_id not in user_dict:
                user_dict[user_id] = user_counter
                user_counter += 1
            user_position = user_dict[user_id]
            if movie_id not in movie_dict:
                movie_dict[movie_id] = movie_counter
                movie_counter += 1
            movie_position = movie_dict[movie_id]
            rating_matrix[user_position, movie_position] = rating

    print "####### Finished parsing #########"
    if shrink_size:
        if shrink_size[0] == rating_matrix[0] and \
                shrink_size[1] == rating_matrix[1]:
            return rating_matrix
        else:
            return rating_matrix[:shrink_size[0], :shrink_size[1]]
    else:
        return rating_matrix

def get_test_set(test_percentage, rating_matrix):
    non_zero_entries = np.array(rating_matrix.nonzero())  # 2 x k
    num_non_zero_entries = non_zero_entries.shape[1]
    num_test = int(num_non_zero_entries * test_percentage)
    random.seed(1)
    test_idx = random.sample(xrange(num_non_zero_entries), num_test)
    return non_zero_entries[:, test_idx]  # 2 x 0.1k

def get_training_matrix(train_usage, test_set, rating_matrix):
    training_matrix = rating_matrix.copy()
    num_test = test_set.shape[1]
    # Set all test entries to zero.
    training_matrix[tuple(test_set)] = 0

    # Choose partial training data according to train_usage.
    non_zero_entries = np.array(training_matrix.nonzero()) # 2 x k
    num_non_zero_entries = non_zero_entries.shape[1]
    random.seed(100)
    num_non_train = int((1.0 - train_usage) * num_non_zero_entries)
    non_train_idx = random.sample(
            xrange(num_non_zero_entries), num_non_train)
    non_train_set = non_zero_entries[:, non_train_idx]
    # Set all non-train entries to zero.
    training_matrix[tuple(non_train_set)] = 0
    return training_matrix

def get_toy_dataset(num_row, num_col, rank, fill_rate, bias_flag = 0, noise_level=0):
    np.random.seed(229)
    
    # low rank matrix
    A = np.random.randn(num_row, rank)
    B = np.random.randn(rank, num_col)
    C = A.dot(B)
    
    # User and item bias
    if bias_flag != 0:
        x = np.random.randn(num_row, 1)
        y = np.random.randn(1, num_col)
        C += x * np.ones((1,num_col)) + np.ones((num_row,1)) * y
    
    # Random noise
    if noise_level > 0:
        C += np.random.randn(num_row, num_col) * np.sqrt(noise_level)
    
    # partial observation
    E = (np.random.rand(num_row, num_col) > fill_rate)
    C[E] = 0
    C = lil_matrix(C)
    return C

def read_from_matrix(filepath, shrink_size):
    raw_matrix = mmread(filepath)
    raw_matrix = raw_matrix.tolil()
    if shrink_size:
        if shrink_size[0] == raw_matrix[0] and \
                shrink_size[1] == raw_matrix[1]:
            return raw_matrix
        else:
            return raw_matrix[:shrink_size[0], :shrink_size[1]]
    else:
        return raw_matrix


# Sample usage.
# data_dir = '../data/'
# filename = 'ratings.csv'

# rating_matrix = rating_matrixparse_user_rating_file(data_dir+filename)

# out_filename = 'ratings.mtx'
# print "######## Saving to disk ########"
# mmwrite(data_dir + out_filename, rating_matrix)
# # To read this matrix, use the following command.
# # A = mmread('ratings.mtx')



