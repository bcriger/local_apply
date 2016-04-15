import numpy as np

def local_apply(rho, choi, qubit):
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
    n = int(np.rint(np.log2(rho.shape[-1])))
    
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

    return rho

def top_apply(mat, choi):
    """
    Treats the matrix mat (which may not be a proper density matrix) as
    a 2-by-2 matrix which is matrix-valued.

    Warning: this function does not make sense. This is because matrix 
    multiplication in numpy is weird, you iterate over the last index 
    of the left matrix and the second-to-last of the right matrix. This
    makes the idea of a matrix-valued matrix unnatural.  
    """
    output = np.zeros_like(mat)
    l = mat.shape[0] / 2 #should work, haha
    
    for r in range(2):
        for c in range(2):
            output += np.kron(choi[r, c], mat[r * l : (r + 1) * l, c * l : (c + 1) * l])

    return output

def unit_apply(rho, choi):
    sz = rho.shape[0]
    output = np.zeros((sz, sz), rho.dtype)
    
    for r in range(2):
        for c in range(2):
            output += rho[r, c] * choi[r, c]

    return output
