'''
Created on 29/03/2013

@author: Mads
'''
import cython
import math


def cycheck(p):
    for i in range(10):
        for y in range(2, int(math.sqrt(p)) + 1):
            if p % y == 0:
                return False
    return True

@cython.ccall
@cython.locals(y=cython.int, p=cython.ulonglong)
def cycheck_pure(p):
    for i in range(10):
        for y in range(2, int(math.sqrt(p)) + 1):
            if p % y == 0:
                return False
    return True


def cycheck_cdef(p):  #cpdef cycheck_cdef(unsigned long long p):
    #cdef int y
    for i in range(10):
        for y in range(2, int(math.sqrt(p)) + 1):
            if p % y == 0:
                return False
    return True
