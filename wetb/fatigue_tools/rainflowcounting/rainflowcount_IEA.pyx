from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import numpy as np
import cython

"""Rainflow counting algorithms
 
These routines are implemented as described in
"Recommended Practices for Wind Turbine Testing - 3. Fatigue Loads", 2. edition 1990, Appendix A

peak_trough: Extract extreme values (local minimum/maximum)
pair_range: list of half-cycle-amplitudes
pair_range_from_to: list of half-cycle from-to values
pair_range_amplitude_mean:  list of half-cycle-amplitudes and mean values
"""


@cython.locals(BEGIN=cython.int, MINZO=cython.int, MAXZO=cython.int, ENDZO=cython.int,
               R=cython.int, L=cython.int, i=cython.int, p=cython.int, f=cython.int)
def peak_trough(x, R):  # cpdef np.ndarray[long,ndim=1] peak_trough(np.ndarray[long,ndim=1] x, int R):
    """
    Returns list of local maxima/minima.

    x: 1-dimensional numpy array containing signal
    R: Thresshold (minimum difference between succeeding min and max

    This routine is implemented directly as described in
    "Recommended Practices for Wind Turbine Testing - 3. Fatigue Loads", 2. edition 1990, Appendix A
    """

    BEGIN = 0
    MINZO = 1
    MAXZO = 2
    ENDZO = 3
    S = np.zeros(x.shape[0] + 1, dtype=np.int)

    L = x.shape[0]
    goto = BEGIN

    while 1:
        if goto == BEGIN:
            trough = x[0]
            peak = x[0]

            i = 0
            p = 1
            f = 0
            while goto == BEGIN:
                i += 1
                if i == L:
                    goto = ENDZO
                    continue
                else:
                    if x[i] > peak:
                        peak = x[i]
                        if peak - trough >= R:
                            S[p] = trough
                            goto = MAXZO
                            continue
                    elif x[i] < trough:
                        trough = x[i]
                        if peak - trough >= R:
                            S[p] = peak
                            goto = MINZO
                            continue

        elif goto == MINZO:
            f = -1

            while goto == MINZO:
                i += 1
                if i == L:
                    goto = ENDZO
                    continue
                else:
                    if x[i] < trough:
                        trough = x[i]
                    else:
                        if x[i] - trough >= R:
                            p += 1
                            S[p] = trough
                            peak = x[i]
                            goto = MAXZO
                            continue
        elif goto == MAXZO:
            f = 1
            while goto == MAXZO:
                i += 1
                if i == L:
                    goto = ENDZO
                    continue
                else:
                    if x[i] > peak:
                        peak = x[i]
                    else:
                        if peak - x[i] >= R:
                            p += 1
                            S[p] = peak
                            trough = x[i]
                            goto = MINZO
                            continue
        elif goto == ENDZO:

            n = p + 1
            if abs(f) == 1:
                if f == 1:
                    S[n] = peak
                else:
                    S[n] = trough
            else:
                S[n] = (trough + peak) / 2
            S = S[1:n + 1]
            return S


@cython.locals(p=cython.int, q=cython.int, f=cython.int, flow=list, k=cython.int, n=cython.int, ptr=cython.int)
def pair_range_amplitude(x):  # cpdef pair_range(np.ndarray[long,ndim=1]  x):
    """
    Returns a list of half-cycle-amplitudes
    x: Peak-Trough sequence (integer list of local minima and maxima)

    This routine is implemented according to
    "Recommended Practices for Wind Turbine Testing - 3. Fatigue Loads", 2. edition 1990, Appendix A
    except that a list of half-cycle-amplitudes are returned instead of a from_level-to_level-matrix
    """

    x = x - np.min(x)
    k = np.max(x)
    n = x.shape[0]
    S = np.zeros(n + 1)

    #A = np.zeros(k+1)
    flow = []
    S[1] = x[0]
    ptr = 1
    p = 1
    q = 1
    f = 0
    # phase 1
    while True:
        p += 1
        q += 1

        # read
        S[p] = x[ptr]
        ptr += 1

        if q == n:
            f = 1
        while p >= 4:
            if (S[p - 2] > S[p - 3] and S[p - 1] >= S[p - 3] and S[p] >= S[p - 2]) \
                or\
                    (S[p - 2] < S[p - 3] and S[p - 1] <= S[p - 3] and S[p] <= S[p - 2]):
                ampl = abs(S[p - 2] - S[p - 1])
                # A[ampl]+=2 #Two half cycles
                flow.append(ampl)
                flow.append(ampl)
                S[p - 2] = S[p]

                p -= 2
            else:
                break

        if f == 0:
            pass
        else:
            break
    # phase 2
    q = 0
    while True:
        q += 1
        if p == q:
            break
        else:
            ampl = abs(S[q + 1] - S[q])
            # A[ampl]+=1
            flow.append(ampl)
    return flow


@cython.locals(p=cython.int, q=cython.int, f=cython.int, flow=list, k=cython.int, n=cython.int, ptr=cython.int)
def pair_range_from_to(x):  # cpdef pair_range_from_to(np.ndarray[long,ndim=1]  x):
    """
    Returns a list of half-cycle from-to values
    x: Peak-Trough sequence (integer list of local minima and maxima)

    This routine is implemented according to
    "Recommended Practices for Wind Turbine Testing - 3. Fatigue Loads", 2. edition 1990, Appendix A
    except that a list of half-cycle-amplitudes are returned instead of a from_level-to_level-matrix
    """

    x = x - np.min(x)
    k = np.max(x)
    n = x.shape[0]
    S = np.zeros(n + 1)

    A = np.zeros((k + 1, k + 1))
    S[1] = x[0]
    ptr = 1
    p = 1
    q = 1
    f = 0
    # phase 1
    while True:
        p += 1
        q += 1

        # read
        S[p] = x[ptr]
        ptr += 1

        if q == n:
            f = 1
        while p >= 4:
            # print S[p - 3:p + 1]
            # print S[p - 2], ">", S[p - 3], ", ", S[p - 1], ">=", S[p - 3], ", ", S[p], ">=", S[p - 2], (S[p - 2] > S[p - 3] and S[p - 1] >= S[p - 3] and S[p] >= S[p - 2])
            # print S[p - 2], "<", S[p - 3], ", ", S[p - 1], "<=", S[p - 3], ", ", S[p], "<=", S[p - 2], (S[p - 2] < S[p - 3] and S[p - 1] <= S[p - 3] and S[p] <= S[p - 2])
            #print (S[p - 2] > S[p - 3] and S[p - 1] >= S[p - 3] and S[p] >= S[p - 2]) or (S[p - 2] < S[p - 3] and S[p - 1] <= S[p - 3] and S[p] <= S[p - 2])
            if (S[p - 2] > S[p - 3] and S[p - 1] >= S[p - 3] and S[p] >= S[p - 2]) or \
               (S[p - 2] < S[p - 3] and S[p - 1] <= S[p - 3] and S[p] <= S[p - 2]):
                A[S[p - 2], S[p - 1]] += 1
                A[S[p - 1], S[p - 2]] += 1
                S[p - 2] = S[p]
                p -= 2
            else:
                break

        if f == 1:
            break  # q==n
    # phase 2
    q = 0
    while True:
        q += 1
        if p == q:
            break
        else:
            # print S[q], "to", S[q + 1]
            A[S[q], S[q + 1]] += 1
    return A


@cython.locals(p=cython.int, q=cython.int, f=cython.int, flow=list, k=cython.int, n=cython.int, ptr=cython.int)
def pair_range_amplitude_mean(x):  # cpdef pair_range(np.ndarray[long,ndim=1]  x):
    """
    Returns a list of half-cycle-amplitudes and mean values
    x: Peak-Trough sequence (integer list of local minima and maxima)

    This routine is implemented according to
    "Recommended Practices for Wind Turbine Testing - 3. Fatigue Loads", 2. edition 1990, Appendix A
    except that a list of half-cycle-amplitudes are returned instead of a from_level-to_level-matrix
    """

    x = x - np.min(x)
    k = np.max(x)
    n = x.shape[0]
    S = np.zeros(n + 1)
    ampl_mean = []
    A = np.zeros((k + 1, k + 1))
    S[1] = x[0]
    ptr = 1
    p = 1
    q = 1
    f = 0
    # phase 1
    while True:
        p += 1
        q += 1

        # read
        S[p] = x[ptr]
        ptr += 1

        if q == n:
            f = 1
        while p >= 4:
            if (S[p - 2] > S[p - 3] and S[p - 1] >= S[p - 3] and S[p] >= S[p - 2]) \
                or\
                    (S[p - 2] < S[p - 3] and S[p - 1] <= S[p - 3] and S[p] <= S[p - 2]):
                # Extract two intermediate half cycles
                ampl = abs(S[p - 2] - S[p - 1])
                mean = (S[p - 2] + S[p - 1]) / 2
                ampl_mean.append((ampl, mean))
                ampl_mean.append((ampl, mean))

                S[p - 2] = S[p]

                p -= 2
            else:
                break

        if f == 0:
            pass
        else:
            break
    # phase 2
    q = 0
    while True:
        q += 1
        if p == q:
            break
        else:
            ampl = abs(S[q + 1] - S[q])
            mean = (S[q + 1] + S[q]) / 2
            ampl_mean.append((ampl, mean))
    return ampl_mean
