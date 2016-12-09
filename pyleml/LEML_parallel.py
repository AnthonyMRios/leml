import sys

import numpy as np
import scipy.sparse as sparse

from _psparse import pmultiply

class LEMLp:
    def __init__(self, num_factors = 128, num_iterations = 25, reg_param = 1.,
                 stopping_criteria = 1e-3, cg_max_iter = 25, cg_gtol = 1e-3, verbose = False):
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param
        self.cg_max_iter = cg_max_iter
        self.cg_gtol = cg_gtol
        self.verbose = verbose

    def fit(self, train_data, train_labels):
        if not isinstance(train_data,sparse.csc_matrix):
            print 'Scipy sparse CSC Matricies (X, Y) must be used for Parallel LEML' 
            sys.exit(0)
        if not isinstance(train_labels,sparse.csc_matrix):
            print 'Scipy sparse CSC Matricies (X, Y) must be used for Parallel LEML' 
            sys.exit(0)
        
        self.W = np.array(np.random.random((train_data.shape[1], self.num_factors)))
        self.H = np.array(np.random.random((train_labels.shape[1], self.num_factors)))

        prev_loss = None
        for iteration in range(self.num_iterations):
            self.fit_H(train_data, train_labels)
            num_cg_iters = self.fit_W(train_data, train_labels)
            if self.verbose:
                print 'Iteration %d done' % (iteration+1)

    def predict(self, test_data):
        return test_data.dot(self.W).dot(self.H.T)

    def fit_H(self, train_data, train_labels):
        X = pmultiply(train_data, np.asfortranarray(self.W))
        #assert np.all(pmultiply(train_data, np.asfortranarray(self.W)) == train_data.dot(self.W))
        X2 = X.T.dot(X)
        eye_reg_param = np.eye(X2.shape[0])*self.reg_param
        X2 = X2 + eye_reg_param
        inv = np.linalg.inv(X2)
        missing = train_labels.T.dot(X)
        #missing = pmultiply(train_labels.T, np.array(X, order='F'))
        for j in range(train_labels.shape[1]):
            self.H[j,:] =  inv.dot(missing[j,:].flatten()).flatten()

    def fit_W(self, train_data, train_labels):
        def vec(A):
            return A.flatten('F')

        def dloss(w, X, Y, H, reg_param):
            #A = X.dot(W)
            A = pmultiply(X, np.asfortranarray(self.W))
            #assert np.all(pmultiply(X, np.asfortranarray(self.W)) == X.dot(self.W))
            B = Y.dot(H)
            M = H.T.dot(H)
            #return vec(X.T.dot(A.dot(M)-B)) + reg_param*w
            #print 'a1'
            #print X.getformat(), X.T.getformat()
            #assert np.all(pmultiply(X.T.tocsc(), np.asfortranarray(A.dot(M)-B)) == X.T.dot(A.dot(M)-B))
            return vec(pmultiply(X.T.tocsc(), np.asfortranarray(A.dot(M)-B))) + reg_param*w

        self.M = np.dot(self.H.T, self.H)
        def Hs(s, X, reg_param):
            S = s.reshape((X.shape[1],self.H.shape[1]), order='F')
            A = pmultiply(X, np.asfortranarray(S))
            #assert np.all(pmultiply(X, np.asfortranarray(S)) == X.dot(S))
            
            #return vec(X.T.dot(A.dot(self.M))) + reg_param*s
            return vec(pmultiply(X.T.tocsc(), np.asfortranarray(A.dot(self.M)))) + reg_param*s

        wt = vec(self.W)
        rt = -dloss(wt, train_data, train_labels, self.H, self.reg_param)
        dt = rt
        total_iters = 0
        for i in range(self.cg_max_iter):
            if np.linalg.norm(rt) < self.cg_gtol:
                break
            total_iters += 1
            hst = Hs(dt, train_data, self.reg_param)
            rtdot = rt.T.dot(rt)
            at = rtdot/(dt.T.dot(hst))
            wt = wt + at*dt
            rtp1 = rt - at*hst
            bt = rtp1.T.dot(rtp1)/(rtdot)
            rt = rtp1
            dt = rt + bt*dt

        self.W = wt.reshape((self.W.shape[0], self.W.shape[1]), order='F')

        return total_iters
