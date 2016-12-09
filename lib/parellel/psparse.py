from __future__ import division
import os
import time
import numpy as np
import scipy.sparse
from _psparse import pmultiply

n_trials = 10
N, M, P = 1000, 10000, 100
RHO = 0.1

X = scipy.sparse.rand(N, M, RHO).tocsc()
W = np.asfortranarray(np.random.randn(M, P))

assert np.all(pmultiply(X, W) == X.dot(W))

t0 = time.time()
for i in range(n_trials):
    A = pmultiply(X, W)

t1 = time.time()
for i in range(n_trials):
    B = X.dot(W)

t2 = time.time()


print 'This Code : %.5fs' % ((t1 - t0) / n_trials)
print 'Scipy     : %.5fs' % ((t2 - t1) / n_trials)