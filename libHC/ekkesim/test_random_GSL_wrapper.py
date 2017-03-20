""" get some values from a GNU scientific library random number generator via ctypes. CK"""
import ctypes as ct
import numpy as np
from matplotlib.pyplot import *


lib = ct.CDLL("test_random_GSL.so")


N = 100
seeds = [1,2,3,4,5,5,5,5,5,5,5,6,7,8,9]
output = np.zeros((len(seeds),N))
for i,seed in enumerate(seeds):
    buff = np.zeros(N,dtype=np.double)
    ct_buff = buff.ctypes.data_as(ct.POINTER(ct.c_double))
    lib.rantest(seed,N,ct_buff)
    output[i,:] = buff[:]


imshow(output,interpolation='nearest',aspect='auto')
savefig("bla.pdf")
