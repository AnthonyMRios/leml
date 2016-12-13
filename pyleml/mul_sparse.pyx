cimport cython
cimport numpy as np
import numpy as np
from cython.parallel import prange

ctypedef struct cs:
    # matrix in compressed-column or triplet form
    int nzmax       # maximum number of entries
    int m           # number of rows
    int n           # number of columns
    int *p          # column pointers (size n+1) or col indices (size nzmax)
    int *i          # row indices, size nzmax
    double *x       # numerical values, size nzmax
    int nz          # # of entries in triplet matrix, -1 for compressed-col

cdef extern int cs_gaxpy(cs *A, double *x, double *y) nogil
cdef extern int cs_gaxpy_row(cs *A, double *x, double *y) nogil

 
@cython.boundscheck(False)
def mul_sparse(sparse not None, np.ndarray[ndim=2, dtype=double, mode='fortran'] dense not None):
    cdef int num_rows = dense.shape[0]
    cdef int num_cols = dense.shape[1]
    cdef np.ndarray[double, ndim=2, mode='fortran'] result
    result = np.zeros((sparse.shape[0], dense.shape[1]), order='F', dtype=np.double)


    cdef np.ndarray[int, ndim=1, mode='c'] indptr  = sparse.indptr
    cdef np.ndarray[int, ndim=1, mode='c'] indices = sparse.indices
    cdef np.ndarray[double, ndim=1, mode='c'] data = sparse.data

    cdef int sparse_num_rows = indptr.size - 1
    cdef int sparse_num_cols = sparse.shape[1]
    cdef int i, j, begin_col_index, end_col_index, index
    cdef double value
    cdef int k
    
    cdef cs csX
    csX.nzmax = sparse.data.shape[0]
    csX.m = sparse.shape[0]
    csX.n = sparse.shape[1]
    csX.p = &indptr[0]
    csX.i = &indices[0]
    csX.x = &data[0]
    csX.nz = -1  # to indicate CSC format

    # Iterate over all non-zero columns in this row
    for k in prange(num_cols, nogil=True):
        cs_gaxpy(&csX, &dense[0, k], &result[0, k])
    '''
    for k in range(num_cols):
        for i in range(sparse_num_cols):
            for index in range(indptr[i], indptr[i+1]):
                result[indices[index],k] += data[indices[index]]*dense[i,k]
    '''
    return result

@cython.boundscheck(False)
def mul_sparse_row(sparse not None, np.ndarray[ndim=2, dtype=double, mode='fortran'] dense not None):
    cdef int num_rows = dense.shape[0]
    cdef int num_cols = dense.shape[1]
    cdef np.ndarray[double, ndim=2, mode='fortran'] result
    result = np.zeros((sparse.shape[0], dense.shape[1]), order='F', dtype=np.double)


    cdef np.ndarray[int, ndim=1, mode='c'] indptr  = sparse.indptr
    cdef np.ndarray[int, ndim=1, mode='c'] indices = sparse.indices
    cdef np.ndarray[double, ndim=1, mode='c'] data = sparse.data

    cdef int sparse_num_rows = indptr.size - 1
    cdef int sparse_num_cols = sparse.shape[1]
    cdef int i, j, begin_col_index, end_col_index, index
    cdef double value
    cdef int k
    
    cdef cs csX
    csX.nzmax = sparse.data.shape[0]
    csX.m = sparse.shape[0]
    csX.n = sparse.shape[1]
    csX.p = &indptr[0]
    csX.i = &indices[0]
    csX.x = &data[0]
    csX.nz = -2  # to indicate CSR format

    # Iterate over all non-zero columns in this row
    for k in prange(num_cols, nogil=True):
        cs_gaxpy_row(&csX, &dense[0, k], &result[0, k])
    return result
