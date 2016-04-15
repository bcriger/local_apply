cimport numpy as np
import numpy as np
cimport cython
DTYPE = np.complex128
ctypedef np.complex128_t DTYPE_t

@cython.boundscheck(True)
cpdef np.ndarray[DTYPE_t, ndim=2] local_apply(np.ndarray[DTYPE_t, ndim=2] rho, np.ndarray[DTYPE_t, ndim=4] choi, int qubit):
    """
    DISCLAIMER: This code was written quickly and not tested. If you 
    experience a problem, it could easily be my fault.

    This code is intended to apply a single-qubit CPTP map to a 
    multi-qubit density matrix without copying the 
    whole matrix at once. You input the density matrix, the Choi
    matrix and the qubit where the channel is to be applied.

    The density matrix is a numpy ndarray or matrix.
    The Choi matrix is a 4D array whose (i, j, k, l)-th entry is the
    (k, l)-th element of the action of the channel on | i >< j |. 
    The qubit is an integer from 0 to n-1.
    """
    cdef unsigned int n = int(np.rint(np.log2(rho.shape[0])))
    cdef unsigned int b_sz, b_r, b_c
    #three cases, top bottom, and middle
    if qubit == 0:
        rho = top_apply(rho, choi)
    
    elif qubit == n - 1:
        for b_r in range(2**(n - 1)):
            for b_c in range(2**(n - 1)):
                rho[2 * b_r : 2 * b_r + 2, 2 * b_c : 2 * b_c + 2] = \
                    unit_apply(rho[2 * b_r : 2 * b_r + 2, 2 * b_c : 2 * b_c + 2], choi)
    
    else: 
    
        b_sz = 2 ** (n - qubit)
    
        for b_r in range(2 ** (qubit - 1)):
            for b_c in range(2 ** (qubit - 1)):
                rho[b_r * b_sz : (b_r + 1) * b_sz, b_c * b_sz : (b_c + 1) * b_sz] = \
                    top_apply(rho[b_r * b_sz : (b_r + 1) * b_sz, b_c * b_sz : (b_c + 1) * b_sz], 
                                choi)

    pass #return rho

@cython.boundscheck(True)
cpdef np.ndarray[DTYPE_t, ndim=2] top_apply(np.ndarray[DTYPE_t, ndim=2] mat,
                np.ndarray[DTYPE_t, ndim=4] choi):
    """
    Treats the matrix mat (which may not be a proper density matrix) as
    a 2-by-2 matrix which is matrix-valued.

    Warning: this function does not make sense. This is because matrix 
    multiplication in numpy is weird, you iterate over the last index 
    of the left matrix and the second-to-last of the right matrix. This
    makes the idea of a matrix-valued matrix unnatural.  
    """
    cdef unsigned int r, c
    cdef np.ndarray[DTYPE_t, ndim=2] output = np.zeros_like(mat)
    cdef unsigned int l = mat.shape[0] / 2 #should work, haha
    
    # print l, mat.shape[0]
    for r in range(2):
        for c in range(2):
            output += np.kron(choi[r, c], mat[r * l : (r + 1) * l, c * l : (c + 1) * l])

    return output

@cython.boundscheck(True)
cpdef np.ndarray[DTYPE_t, ndim=2] unit_apply(np.ndarray[DTYPE_t, ndim=2] rho,
               np.ndarray[DTYPE_t, ndim=4] choi):
    cdef unsigned int r, c
    cdef unsigned int sz = rho.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] output = np.zeros((sz, sz), DTYPE)
    
    for r in range(2):
        for c in range(2):
            output += rho[r, c] * choi[r, c]

    return output
