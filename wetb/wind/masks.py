'''
Created on 06/11/2015

@author: MMPE
'''


def wsp_mask(x, wsp, pm_tolerance):
    return (wsp - pm_tolerance <= x) & (x < wsp + pm_tolerance)

def wdir_mask(x, wdir, pm_tolerance):
    return ((x - wdir + pm_tolerance) % 360 >= 0) & ((x - wdir + pm_tolerance) % 360 < 2 * pm_tolerance)

