# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 07:29:24 2016

@author: Yuan Chen
"""
import copy
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import numpy as np
import matplotlib.pyplot as plt
# A few parameters
#global gamma # How much we change the step
global reg # The parameter to keep singular values of A small.
global k # This is the number of singular values to compute (for sparse matrices). The default value of splinalg.svds is 6.
global rank
reg = 0.5
k = 10
rank = 5
#gamma = 1.2
# The main function of Algorithm 1 in [1]
def ExtendedGradientAlgorithm(W0,L0,gamma,niter,testSet):
    global rows,columns,values,trainingSize
#    global testRows,testColumns,testValues,testSize
    rows,columns,values = sp.find(W0) # Training set
#    testRows = W1[0,:] # Test set
#    testColumns = W1[1,:]
#    testSize = len(testRows)
#    testValues = np.reshape(testValue,testSize)
 #   testRows = [x for (x,y) in testIndex]
  #  testColumns = [y for (x,y) in testIndex]
    trainingSize = len(rows)
    trainingError = []
    testError = []
    # Initialize variables    
    L = L0
    W_new = W0.copy()
    u,s,vt = splinalg.svds(W_new,rank)
    print 'Initialization Done!\n'
    # Begin iteration
    for l in range(niter):
        u_prev,s_prev,vt_prev,W_old = u,s,vt,W_new
        u,s,vt = _p(L,W_old)
        W_new = u.dot(np.diag(s)).dot(vt)
   #     W_new = u.dot(np.diag(s)).dot(vt)
   #     while(_fsvds(u,s,vt) + reg*sum(s) > _Q(L,u,s,vt,W_new,u_prev,s_prev,vt_prev,W_old)):
#        while((_fsvds(u,s,vt) + reg*sum(s)) > _Q(L,u,s,vt,u_prev,s_prev,vt_prev,W_old)):
        while(_fsvds(u,s,vt) > _Q(L,W_old,W_new)):
            print 'L='+str(L)+'\n'
            print _fsvds(u,s,vt)
            print _Q(L,W_new,W_old)
            L = gamma*L
            u,s,vt = _p(L,W_old)
            W_new = u.dot(np.diag(s)).dot(vt)
        print 'L='+str(L)
       # W_new = u.dot(np.diag(s)).dot(vt)
       # print 'F='+str(_f(W_new)+reg*sum(s))
        trainingError.append(_f(W_new)/trainingSize)
   #     testError.append(_getTestError(W_new)/testSize)
        print "training error this step:"+str(trainingError[l])+'\n'
    #    print "test error this step:"+str(testError[l])+'\n'
        print "Done for "+str(l)+"-th iteration!\n"
    # Plot them
    to_return = np.maximum(np.minimum(5,W_new[tuple(testSet)]),0)
    
    return trainingError,to_return
#    plt.figure()
#    x = range(1,1+niter) 
#    plt.plot(x,trainingError,'g',x,testError,'r')
#    plt.show()
# The main function of Algorithm 2 in [1]
def AcceleratedGradientAlgorithm(W0,L0,gamma,niter,W1):
    global rows,columns,values,trainingSize
    global testRows,testColumns,testValues,testSize
    rows,columns,values = sp.find(W0) # Used in training error
    testRows,testColumns,testValues = sp.find(W1)
    trainingSize = len(rows)
    testSize = len(testRows)
    L = L0
    trainingError = []
    testError = []
    W_new = copy.deepcopy(W0)
    Z_new = W_new
    u,s,vt = splinalg.svds(Z_new,rank)
    alpha_new = 1
    print 'Initialization Done!\n'
    for k in range(niter):
        u_prev,s_prev,vt_prev,Z_old,W_old,alpha_old = u,s,vt,Z_new,W_new,alpha_new
        u,s,vt = _p(L,Z_old)
       # Z_new = u.dot(np.diag(s)).dot(vt)
        while((_fsvds(u,s,vt) + reg*sum(s)) > _Q(L,u,s,vt,u_prev,s_prev,vt_prev,Z_old)):
            L = gamma*L
            u,s,vt = _p(L,Z_old)
        Z_new = u.dot(np.diag(s)).dot(vt) # This step may be optimized because at the moment we don't need full W_new
        W_new = _p(L,Z_old)
        alpha_new = 0.5*(1+np.sqrt(1+4*alpha_old**2))
        Z_new = W_new + (alpha_old - 1.0)/alpha_new*(W_new - W_old)
        trainingError.append(_f(W_new))
        testError.append(_getTestError(W_new))
        print "Done for "+str(l)+"-th iteration!\n"
    # Plot them
    plt.figure()
    x = range(niter) + 1
    plt.plot(x,trainingError/trainingSize,'g',x,testError/testSize,'r')
    plt.show()
# The cost function to minimize
def _f(W):
    return sum([(W[rows[i],columns[i]] - values[i])**2 for i in range(trainingSize)])
def _fsvds(u,s,vt):
   # print s
    S = np.diag(s)
#    print type(u)
 #   print type(vt)
    model_value = [(u[rows[i],:]).dot(S).dot(vt[:,columns[i]]) for i in range(trainingSize)]
    return sum([(model_value[i]-values[i])**2 for i in range(trainingSize)])
# This function calculates argmin_X Q_miu (X,Y). (equation 11 in [1]. Also see equation 9 and 10 in [1])
def _p(L,Z):
    Y = Z.copy()
    if (sp.issparse(Y)):
        Y = Y.toarray()
    for i in range(trainingSize):
        Y[rows[i],columns[i]] *= (1-2.0/L)
        Y[rows[i],columns[i]] += 2.0/L*values[i]
#    return Y
    [u,s,vt] = np.linalg.svd(Y,full_matrices=False)
    s = (np.maximum(s - reg/L,0))[0:rank]
    u = u[:,0:rank]
    vt = vt[0:rank,:]
    return u,s,vt
 #The auxilliary function Q_miu(X,Y)
#def _Q(L,u,s,vt,u_prev,s_prev,vt_prev,W_old):
def _Q(L,W_old,W_new):
    # This part may be optimized further
#    return (_fsvds(u_prev,s_prev,vt_prev) + _matrDot(u,s,vt,W_old)
#           + L/2.0*(s.dot(s)+s_prev.dot(s_prev)
#           - 2*np.trace(np.transpose(vt_prev).dot((np.diag(s_prev)).dot(np.transpose(u_prev).dot(u)).dot(np.diag(s)).dot(vt))))
#           + reg*sum(s))
    return _f(W_old)+_matrDot(W_new,W_old)+0.5*L*(np.linalg.norm(W_new-W_old,'fro')**2)
#def _matrDot(W_new,W_old):
#    to_return = 0
#    for i in range(trainingSize):
#        to_return += 2*(W_new[rows[i],columns[i]] - W_old[rows(i),columns[i]])*(W_old[rows(i),columns[i]] - values[i])
#    return to_return
#def _matrDot(u,s,vt,W_old):
def _matrDot(W_new,W_old):
    to_return = 0
    for i in range(trainingSize):
 #       to_return += 2*(_getW(u,s,vt,i) - W_old[rows[i],columns[i]])*(W_old[rows[i],columns[i]] - values[i])
        to_return += 2*(W_new[rows[i],columns[i]] - W_old[rows[i],columns[i]])*(W_old[rows[i],columns[i]] - values[i])
    return to_return
def _getTestError(W):
  #  print 'testRows'+str(testRows)+'\n'
  #  print 'testColumns'+str(testColumns)+'\n'
  #  print 'testValues'+str(testValues)+'\n'
    return sum([(W[testRows[i],testColumns[i]] - testValues[i])**2 for i in range(testSize)])
def _getW(u,s,vt,i):
    S = np.diag(s)
    return u[rows[i],:].dot(S).dot(vt[:,columns[i]])
