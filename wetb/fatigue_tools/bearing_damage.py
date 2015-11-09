'''
Created on 13/10/2014

@author: MMPE
'''

def bearing_damage_SWP(angle_moment_lst, m=3, thresshold=0.1):
    """Function ported from Matlab.

    Matlab documentation:
    "Compute bearing damage according to SWP procedure
    The basic equation is a damage equivalent loading calculated as

    Damage = sum(dPitch*Mr^3.0)

    Where
    dPitch = pitch change in a given timestep [deg]
    Mr = load component in a given sector [kNm]

    We evaluate Mr (blade root moment resultant) in 10deg sectors and take
    the largest damage. For a delta loading as you are investigating it is
    accurate enough just to take the flap loading and ignore the changes
    coming from edge load variation.

    The delta pitch is a bit tricky. You need to filter you signal so minor
    quick variation in pitch does not give an actual delta pitch. The
    variation in pitch reference is filtered. So effectively a variation
    below 0.1deg is ignored and the pitch is fixed with no variation as
    long as the change is below this value.

    So basically for each time step calulate dPitch*Mr^3 and sum it up with
    Mr=blade flap moment. I’m sure your model will not have the details for
    capacity comparison, but for changes from one configuration to another
    should be ok. We do not use the sum(dPitch) for design evaluation directly."

    Parameters
    ----------
    angle_moment_lst : ((angle_A_vector, moment_A_vector),(angle_B_vector, moment_B_vector),...)
        Angles[deg] and momements (e.g. pitch angle and blade root moment for the blades of a wind turbine)
    m : int, optional
        analogue to Wöhler exponent, should be 3
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

