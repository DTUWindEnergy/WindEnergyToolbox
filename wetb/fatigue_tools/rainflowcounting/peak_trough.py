import cython
import numpy as np
# cimport numpy as np

@cython.locals(BEGIN=cython.int, MINZO=cython.int, MAXZO=cython.int, ENDZO=cython.int, \
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
