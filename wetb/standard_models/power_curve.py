import numpy as np


def standard_power_curve(power_norm, diameter, turbulence_intensity=.1, shear=(0, 100),
                         rho=1.225, max_cp=.49,
                         gear_loss_const=.01, gear_loss_var=.014, generator_loss=0.03, converter_loss=.03):
    """Generate standard power curve

    The method is extracted from an Excel spreadsheet made by Kenneth Thomsen, DTU Windenergy and
    ported into python by Mads M. Pedersen, DTU Windenergy.

    Parameters
    ----------
    power_norm : int or float
        Nominal power [kW]
    diameter : int or float
        Rotor diameter [m]
    turbulence_intensity : float
        Turbulence intensity [%]
    shear : (power shear coefficient, hub height)
        Power shear arguments\n
        - Power shear coeficient, alpha\n
        - Hub height [m]
    rho : float optional
        Density of air [kg/m^3], defualt is 1.225
    max_cp : float
        Maximum power coefficient
    gear_loss_const : float
        Constant gear loss [%]
    gear_loss_var : float
        Variable gear loss [%]
    generator_loss : float
        Generator loss [%]
    converter_loss : float

    Examples
    --------
    wsp, power = standard_power_curve(10000, 178.3)
    plot(wsp, power)
    show()
    """


    area = (diameter / 2) ** 2 * np.pi
    wsp_lst = np.arange(0.5, 25, .5)
    sigma_lst = wsp_lst * turbulence_intensity

    def norm_dist(x, my, sigma):
        if turbulence_intensity > 0:
            return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - my) ** 2 / (2 * sigma ** 2))
        else:
            return x == my
    p_aero = .5 * rho * area * wsp_lst ** 3 * max_cp / 1000

    # calc power - gear, generator and conv loss
#     gear_loss = gear_loss_const * power_norm + gear_loss_var * p_aero
#     p_gear = p_aero - gear_loss
#     p_gear[p_gear < 0] = 0
#     gen_loss = generator_loss * p_gear
#     p_gen = p_gear - gen_loss
#     converter_loss = converter_loss * p_gen
#     p_raw = p_gen - converter_loss
#     p_raw[p_raw > power_norm] = power_norm
    p_raw = p_aero - power_loss(p_aero, power_norm, gear_loss_const, gear_loss_var, generator_loss, converter_loss)

    powers = []
    shear_weighted_wsp = []
    alpha, hub_height = shear
    r = diameter / 2
    z = np.linspace(hub_height - r, hub_height + r, 100, endpoint=True)
    shear_factors = (z / hub_height) ** alpha
    rotor_width = 2 * np.sqrt(r ** 2 - (hub_height - z) ** 2)

    for wsp, sigma in zip(wsp_lst, sigma_lst):
        shear_weighted_wsp.append(wsp * (np.trapz(shear_factors ** 3 * rotor_width, z) / (area)) ** (1 / 3))
        ndist = norm_dist(wsp_lst, wsp, sigma)
        powers.append((ndist * p_raw).sum() / ndist.sum())

    return wsp_lst, lambda wsp : np.interp(wsp, wsp_lst, powers)
    return wsp_lst, np.interp(shear_weighted_wsp, wsp_lst, powers)

def power_loss(power_aero, power_norm, gear_loss_const=.01, gear_loss_var=.014, generator_loss=0.03, converter_loss=.03):
    gear_loss = gear_loss_const * power_norm + gear_loss_var * power_aero
    p_gear = power_aero - gear_loss
    p_gear[p_gear < 0] = 0
    gen_loss = generator_loss * p_gear
    p_gen = p_gear - gen_loss
    converter_loss = converter_loss * p_gen
    p_electric = p_gen - converter_loss
    p_electric[p_electric > power_norm] = power_norm
    return power_aero - p_electric
    

if __name__ == '__main__':


    from matplotlib.pyplot import plot, show
    plot(*standard_power_curve(10000, 178.3, 0., (0, 119), gear_loss_const=.0, gear_loss_var=.0, generator_loss=0.0, converter_loss=.0))
    plot(*standard_power_curve(10000, 178.3, 0.03, (0, 119), gear_loss_const=.0, gear_loss_var=.0, generator_loss=0.0, converter_loss=.0))




    show()
