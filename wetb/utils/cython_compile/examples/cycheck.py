'''
Created on 29/03/2013

@author: Mads
'''
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import range
from builtins import int
from future import standard_library
standard_library.install_aliases()
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
