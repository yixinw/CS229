'''
This util file helps parsing rating data file into a huge sparse matrix.
Each row corresponds to a user, each column corresponds to a movie.

Sample usage:
data_dir = '../data/'
filename = 'ratings.csv'
rating_matrix = rating_matrixparse_user_rating_file(data_dir+filename)

out_filename = 'rating.mtx'
print "######## Saving to disk ########"
mmwrite(data_dir + out_filename, rating_matrix)
# To read this matrix, use the following command.
# A = mmread('rating.mtx')
'''

import csv
from scipy.sparse import dok_matrix
import IPython
import numpy as np
from scipy.io import mmread, mmwrite

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

    rating_matrix = dok_matrix((num_user, num_movie)) # size: num_user x num_movie

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

# Sample usage.
# data_dir = '../data/'
# filename = 'ratings.csv'

# rating_matrix = rating_matrixparse_user_rating_file(data_dir+filename)

# out_filename = 'rating.mtx'
# print "######## Saving to disk ########"
# mmwrite(data_dir + out_filename, rating_matrix)
# # To read this matrix, use the following command.
# # A = mmread('rating.mtx')



