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

def parse_user_rating_file(filename):
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

# Sample usage.
# data_dir = '../data/'
# filename = 'ratings.csv'

# rating_matrix = rating_matrixparse_user_rating_file(data_dir+filename)

# out_filename = 'ratings.mtx'
# print "######## Saving to disk ########"
# mmwrite(data_dir + out_filename, rating_matrix)
# # To read this matrix, use the following command.
# # A = mmread('ratings.mtx')



