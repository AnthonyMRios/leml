import numpy as np
import scipy.sparse as scipy_sp

class LEMLs:
    def __init__(self, num_factors = 128, num_iterations = 25, reg_param = 1.,
                 stopping_criteria = 1e-3, cg_max_iter = 25, cg_gtol = 1e-3, verbose = False):
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param
        self.cg_max_iter = cg_max_iter
        self.cg_gtol = cg_gtol
        self.verbose = verbose

    def fit(self, train_data, train_labels):
        self.W = np.random.random((train_data.shape[1], self.num_factors))
        self.H = np.random.random((train_labels.shape[1], self.num_factors))

        prev_loss = None
        for iteration in range(self.num_iterations):
            self.fit_H(train_data, train_labels)
            num_cg_iters = self.fit_W(train_data, train_labels)
            if self.verbose:
                print 'Iteration %d done' % (iteration+1)

    def predict(self, test_data):
        return test_data.dot(self.W).dot(self.H.T)

    def fit_H(self, train_data, train_labels):
        X = train_data.dot(self.W)
        X2 = X.T.dot(X)
        eye_reg_param = np.eye(X2.shape[0])*self.reg_param
        X2 = X2 + eye_reg_param
        inv = np.linalg.inv(X2)
        missing = train_labels.T.dot(X)
        for j in range(train_labels.shape[1]):
            self.H[j,:] =  inv.dot(missing[j,:].flatten()).flatten()

    def fit_W(self, train_data, train_labels):
        def vec(A):
            return A.flatten('F')

        def dloss(w, X, Y, H, reg_param):
            W = self.W
            A = X.dot(W)
            B = Y.dot(H)
            M = H.T.dot(H)
            return vec(X.T.dot(A.dot(M)-B)) + reg_param*w

        self.M = np.dot(self.H.T, self.H)
        def Hs(s, X, reg_param):
            S = s.reshape((X.shape[1],self.H.shape[1]), order='F')
            A = X.dot(S)
            AdM = A.dot(self.M)
            XdAdM = X.T.dot(AdM)
            v = vec(XdAdM)
            return v + reg_param*s
            #return vec((X.T.dot(A.dot(self.M)))) + reg_param*s

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



import sys
from time import time

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import precision_score 
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

#from pyleml import LEML

def main():
    print 'Loading data'
    sys.stdout.flush()
    X = joblib.load('../example/test_data/bibtex-train.pkl')
    labels = joblib.load('../example/test_data/bibtex-Y-train.pkl')
    X_test = joblib.load('../example/test_data/bibtex-test.pkl')
    labels_test = joblib.load('../example/test_data/bibtex-Y-test.pkl')
    print X.shape, labels.shape, X.getformat(), labels.getformat()

    print 'Training LEML'
    sys.stdout.flush()
    #for l in [1e-3, 1e-2, 1e-1, 1., 10]:
    t0 = time()
    #leml = LEML.get_instance('single', num_factors=200, num_iterations=25, reg_param=1., verbose=True)
    leml = LEMLs(num_factors=200, num_iterations=5, reg_param=1., verbose=True)
    leml.fit(X.tocsr(), labels.tocsr())
    print 'Train time', time() - t0, 'seconds'
    sys.stdout.flush()
    preds = leml.predict(X_test)
    preds_top_k = preds.argsort()[:,::-1]
    preds_top_k = preds_top_k[:,:1]
    new_preds = np.zeros((preds.shape[0], preds.shape[1]))
    new_preds[np.arange(preds.shape[0]).repeat(1),preds_top_k.flatten()] = 1
    print 'Precision @ 1:', precision_score(labels_test.toarray(), new_preds, average='samples')

    preds_top_k = preds.argsort()[:,::-1]
    preds_top_k = preds_top_k[:,:3]
    new_preds = np.zeros((preds.shape[0], preds.shape[1]))
    new_preds[np.arange(preds.shape[0]).repeat(3),preds_top_k.flatten()] = 1
    print 'Precision @ 3:', precision_score(labels_test.toarray(), new_preds, average='samples')

    preds_top_k = preds.argsort()[:,::-1]
    preds_top_k = preds_top_k[:,:5]
    new_preds = np.zeros((preds.shape[0], preds.shape[1]))
    new_preds[np.arange(preds.shape[0]).repeat(5),preds_top_k.flatten()] = 1
    print 'Precision @ 5:', precision_score(labels_test.toarray(), new_preds, average='samples')


if __name__ == '__main__':
    main()
