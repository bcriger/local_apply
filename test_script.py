'''
import pyximport
pyximport.install();
from local_apply import *
'''
from local_apply_debug import *
import numpy as np

gamma = 0.1
n = 10

#amplitude damping
choi_map = np.array([[
    [[1., 0], [0, 0]], [[0, np.sqrt(1. - gamma)], [0, 0]]], 
    [[[0, 0], [np.sqrt(1. - gamma), 0]], [[gamma, 0], [0, 1. - gamma]]
    ]], dtype=np.complex128)

dens_mat = np.zeros((2**n, 2**n), dtype=np.complex128)
dens_mat[2**n - 1, 2**n - 1] = 1.

#print rand_dens_mat
for q in range(n):
    rho = local_apply( dens_mat, choi_map, q)
    #print rho