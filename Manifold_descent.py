
# coding: utf-8

# In[1]:

import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import time


# In[2]:

def trimming(A, penalty):
    [m, n] = A.shape
    [id_row, id_col, val] = ss.find(A)
    num_revealed = len(id_row)

    # remove dense rows
    thres_row = num_revealed * penalty / m
    row_count = np.zeros(m)
    for i in range(num_revealed):
        row_count[id_row[i]] += 1
    for r in range(m):
        if row_count[r] > thres_row:
            A[r, :] = np.zeros([1,n])

    # remove dense columns
    thres_col = num_revealed * penalty / n
    col_count = np.zeros(n)
    for i in range(num_revealed):
        col_count[id_col[i]] += 1
    for c in range(n):
        if col_count[c] > thres_col:
            A[:, c] = np.zeros([m,1])

    return A


# In[3]:

def obj(A, U, V, r):
    [id_row, id_col, val] = ss.find(A)
    num_revealed = len(id_row)

    # solve a linear system
    B = np.zeros([r**2,r**2])

    for k0 in range(r):
        for l0 in range(r):
            row_idx = k0 * r + l0
            temp = ss.lil_matrix(U[:,k0] * V[:,l0].T)
            temp = U.T * temp.multiply(A!=0) * V
            B[row_idx, :] = temp.reshape(1,r**2)

    b = U.T * A * V
    b = b.reshape(r**2, 1)

    S = np.linalg.solve(B,b)
    S = np.asmatrix(S)
    S = S.reshape(r,r)

    # compute the objective
    R = ss.lil_matrix(A - U * S * V.T)
    R = R.multiply(A!=0)

    return [S, (R.multiply(R)).sum()]


# In[4]:

def geodesic_path(U, X, D, Y, t):
    cos_mat = np.asmatrix( np.diag(np.cos(D*t)) )
    sin_mat = np.asmatrix( np.diag(np.sin(D*t)) )
    R = U * Y.T * cos_mat * Y + X * sin_mat * Y
    return R


# In[9]:

def manifold_descent(A, r, max_iter=20, err_thresh=1e-4, penalty=2, alpha=0.5, beta=0.5):
    # trimming
    A = trimming(A, penalty)
    [id_row, id_col, val] = ss.find(A)
    [m, n] = A.shape

    # SVD initialization
    [U, S, V] = ssl.svds(A, r)
    U = np.asmatrix(U)
    S = np.diag(S)
    V = np.asmatrix(V)
    V = V.T
    [S0, cost] = obj(A,U,V,r)
    print "simple SVD done with fitted error: %f." %(np.sqrt(cost)/np.sqrt(len(id_row)))

    # Gradient Descent Iteration
    t = 0.02
    for count in range(max_iter):
        # residue matrix
        R = ss.lil_matrix(A - U * S * V.T)
        R = R.multiply(A!=0)

        # computation of negative gradient
        U_grad = R * V * S.T
        U_grad = U_grad - U * U.T * U_grad
        V_grad = R.T * U * S
        V_grad = V_grad - V * V.T * V_grad

        # backtracking line search
        [X1, D1, Y1] = np.linalg.svd(U_grad, 0)
        X1 = np.asmatrix(X1)
        Y1 = np.asmatrix(Y1)
        [X2, D2, Y2] = np.linalg.svd(V_grad, 0)
        X2 = np.asmatrix(X2)
        Y2 = np.asmatrix(Y2)

        [S_new, cost_new] = obj(A, geodesic_path(U,X1,D1,Y1,t), geodesic_path(V,X2,D2,Y2,t), r)
        while cost - cost_new < alpha * t * (np.linalg.norm(U_grad)**2 + np.linalg.norm(V_grad)**2) and t > 1e-9:
            start = time.time()

            t *= beta
            [S_new, cost_new] = obj(A, geodesic_path(U,X1,D1,Y1,t), geodesic_path(V,X2,D2,Y2,t), r)

            end = time.time()
            print "elapsed time1: %f" %(end-start)

        # update U, V, S
        U = geodesic_path(U,X1,D1,Y1,t)
        V = geodesic_path(V,X2,D2,Y2,t)
        S = S_new
        print "Fitted error after iteration %d is: %f." %(count+1, np.sqrt(cost_new)/np.sqrt(len(id_row)))
        print "Stepsize at iteration %d: %f" %(count+1, t)
        if cost_new < err_thresh:
            break
        cost = cost_new
        t *= 4

    result = U * S * V.T
    return result


# In[12]:

def manifold_descent_simplegrad(A, r, max_iter=30, err_thresh=1e-4, penalty=2, alpha=0.5, beta=0.5):
    # trimming
    A = trimming(A, penalty)
    [id_row, id_col, val] = ss.find(A)
    [m, n] = A.shape

    # SVD initialization
    [U, S, V] = ssl.svds(A, r)
    U = np.asmatrix(U)
    S = np.diag(S)
    V = np.asmatrix(V)
    V = V.T
    [S0, cost] = obj(A,U,V,r)
    print "simple SVD done with fitted error: %f." %(np.sqrt(cost)/np.sqrt(len(id_row)))

    # Gradient Descent Iteration
    t = 0.02
    for count in range(max_iter):
        # residue matrix
        R = ss.lil_matrix(A - U * S * V.T)
        R = R.multiply(A!=0)

        # computation of negative gradient
        U_grad = R * V * S.T
        U_grad = U_grad - U * U.T * U_grad
        V_grad = R.T * U * S
        V_grad = V_grad - V * V.T * V_grad

        # backtracking line search
        [S_new, cost_new] = obj(A, U + U_grad * t, V + V_grad * t, r)
        while cost - cost_new < alpha * t * (np.linalg.norm(U_grad)**2 + np.linalg.norm(V_grad)**2) and t > 1e-9:
            start = time.time()

            t *= beta
            [S_new, cost_new] = obj(A, U + U_grad * t, V + V_grad * t, r)

            end = time.time()
            print "elapsed time1: %f" %(end-start)

        # update U, V, S
        U = U + U_grad * t
        V = V + V_grad * t
        S = S_new
        print "Fitted error after iteration %d is: %f." %(count+1, np.sqrt(cost_new)/np.sqrt(len(id_row)))
        print "Stepsize at iteration %d: %f" %(count+1, t)
        if cost_new < err_thresh:
            break
        cost = cost_new
        t *= 4

    result = U * S * V.T
    return result

