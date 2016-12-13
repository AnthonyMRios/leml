#include "cs.h"
/* y = A*x+y */
int cs_gaxpy(const cs *A, const double *x, double *y)
{
    int p, j, n, *Ap, *Ai ;
    double *Ax ;
    if (!CS_CSC (A) || !x || !y) return (0) ;       /* check inputs */
    n = A->n ; Ap = A->p ; Ai = A->i ; Ax = A->x ;
    for (j = 0 ; j < n ; j++)
    {
        for (p = Ap[j] ; p < Ap[j+1] ; p++)
        {
            y[Ai[p]] += Ax[p] * x[j] ;
        }
    }
    return (1) ;
}


int cs_gaxpy_row(const cs *A, const double *x, double *y)
{
    int p, j, n, *Ap, *Ai , m;
    double *Ax ;
    if (!CS_CSR (A) || !x || !y) return (0) ;       /* check inputs */
    n = A->n ; Ap = A->p ; Ai = A->i ; Ax = A->x ; m = A->m ;
    for (j = 0 ; j < m ; j++)
    {
        for (p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            //y[Ai [p]] += Ax[p] * x[j] ;
            y[j] += Ax[p] * x[Ai[p]];
        }
    }
    return (1) ;
}

