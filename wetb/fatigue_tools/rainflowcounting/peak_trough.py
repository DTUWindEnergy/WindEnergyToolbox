import numpy as np
from numba.core.decorators import njit


@njit(cache=True)  # jit faster than previous cython compiled extension
def peak_trough(x, R):
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
    S = np.full(x.shape[0] + 1, 0)  # np.zeros(...).astype(int) not supported by numba

    L = x.shape[0]
    goto = BEGIN

    while True:
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
