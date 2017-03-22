'''
Created on 19/07/2016

@author: MMPE
'''
'''
Created on 02/11/2015

@author: MMPE
'''

import numpy as np
def _begin_end(x, mask):
    begin = np.where(~mask[:-1] & mask[1:])[0]
    end = np.where(mask[:-1] & ~mask[1:])[0] + 1
    if begin[0] > end[0]:
        x[:end[0]] = x[end[0]]
        end = end[1:]
    if begin[-1] > end[-1]:
        x[begin[-1]:] = x[begin[-1]]
        begin = begin[:-1]
    return begin, end

def replace_by_nan(data, spike_mask):
    data[spike_mask] = np.nan
    return data

def replace_by_mean(x, mask):
    x = x.copy()

    for b, e in zip(*_begin_end(x, mask)):
        x[b + 1:e] = np.mean([x[b], x[e]])
    return x

def replace_by_line(x, mask):
    x = x.copy()
    for b, e in zip(*_begin_end(x, mask)):
        x[b + 1:e] = np.interp(np.arange(b + 1, e), [b, e], [x[b], x[e]])
    return x

def replace_by_polynomial(x, mask, deg=3, no_base_points=12):
    if not np.any(mask):
        return x
    x = x.copy()
    begin, end = _begin_end(x, mask)
    for last_e, b, e, next_b in zip(np.r_[0, end[:-1]], begin, end, np.r_[begin[1:], len(x)]):
        if  b - last_e >= no_base_points and next_b - e >= no_base_points:
            pbegin, pend = max(b - no_base_points, last_e), min(e + no_base_points, next_b)
            inter_points = np.r_[np.arange(pbegin + 1, b + 1), np.arange(e, pend)]
            pol = np.poly1d(np.polyfit(inter_points , x[inter_points], deg))(np.arange(b + 1, e))
            if pol.max() - pol.min() > 2 * (x[inter_points].max() - x[inter_points].min()):
                # use linear interpolation if range is too big
                x[b + 1:e] = np.interp(np.arange(b + 1, e), [b, e], [x[b], x[e]])
            else:
                x[b + 1:e] = pol
        else:
            x[b + 1:e] = np.interp(np.arange(b + 1, e), [b, e], [x[b], x[e]])
    return x

def max_cont_mask_length(mask):
    if not np.any(mask):
        return 0
    begin = np.where(~mask[:-1] & mask[1:])[0]
    end = np.where(mask[:-1] & ~mask[1:])[0] + 1
    no_nan = []
    if begin[0] > end[0]:
        no_nan.append(end[0])
        end = end[1:]
    if begin[-1] > end[-1]:
        no_nan.append(len(mask) - begin[-1] - 1)
        begin = begin[:-1]
    return max(np.r_[no_nan, end - 1 - begin])


