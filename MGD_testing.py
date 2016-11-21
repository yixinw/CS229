
# coding: utf-8

# In[4]:

import numpy as np
import scipy.sparse as ss
from Manifold_descent import *

m = 1000
n = 2000
r = 3
prob_reveal = 0.01
A = np.random.randn(m,r)
B = np.random.randn(r,n)
A = np.asmatrix(A)
B = np.asmatrix(B)
C = A * B
E = (np.random.rand(m,n) < prob_reveal)
for i in range(m):
    for j in range(n):
        if ~E[i,j]:
            C[i,j]=0
C = ss.lil_matrix(C)
print np.linalg.norm(A*B-manifold_descent(C,r))/np.sqrt(m*n)
print np.linalg.norm(A*B-manifold_descent_simplegrad(C,r))/np.sqrt(m*n)


# In[ ]:



