'''
Created on 06/09/2013

@author: Mads M. Pedersen (mmpe@dtu.dk)
'''
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from wetb.hawc2.ascii2bin.ascii2bin import ascii2bin

if __name__=="__main__":
    ascii2bin(r"tests/hawc2ascii.sel", "temp_hawc2ascii.sel")

