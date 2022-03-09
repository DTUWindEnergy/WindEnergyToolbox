'''
Created on 13/10/2014

@author: MMPE
'''

def bearing_damage(angle_moment_lst, m=3, thresshold=0.1):
    """Function ported from Matlab.

    Parameters
    ----------
    angle_moment_lst : ((angle_A_vector, moment_A_vector),(angle_B_vector, moment_B_vector),...)
        Angles[deg] and momements (e.g. pitch angle and blade root moment for the blades of a wind turbine)
    m : int, optional
        analogue to Wohler exponent, should be 3
    threeshold : float, optional
        Pitch noise. Pitch movement below this thresshold is ignored

    Returns
    -------
    max_damage : float
        A damage value of the most damaged pitch bearing. Only suitable for comparison
    """

    damages = []
    for angle, moment in angle_moment_lst:
        angle, moment = angle.tolist(), moment.tolist()
        k1 = 0
        p1, mflap = angle[0], moment[0]
        damage = 0
        for k, (pi, mo) in enumerate(zip(angle[1:], moment[1:]), 1):
            dangle = abs(pi - p1)
            if dangle > thresshold:
                damage += dangle * abs(mflap / (k - k1 + 1)) ** m
                k1 = k
                mflap = mo
                p1 = pi
            else:
                mflap += mo
        damages.append(damage)
    return max(damages)

