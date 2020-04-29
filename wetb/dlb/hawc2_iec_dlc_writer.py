from wetb.hawc2.hawc2_input_writer import HAWC2InputWriter
from wetb.hawc2.tests import test_files
import os
from wetb.dlb.iec64100_1 import DTU_IEC64100_1_Ref_DLB

"""
TODO: delete wind ramp / replace wind section
TODO: set default turb_format = 0
"""


class HAWC2_IEC_DLC_Writer(HAWC2InputWriter):
    def __init__(self, base_htc_file, diameter,
                 time_start=100,  # Minimum 5s cf. IEC61400-1(2005), section 7.5
                 turbulence_defaults=(29.4, 3.7, 8192, 32)  # L, gamma, n_x, n_yz):
                 ):
        HAWC2InputWriter.__init__(self, base_htc_file, diameter=diameter,
                                  time_start=time_start,
                                  turbulence_defaults=turbulence_defaults)

    def set_V_hub(self, htc, V_hub, **_):
        htc.wind.wsp = V_hub
        htc.wind.wind_ramp_factor = 0, self.time_start, 8 / V_hub, 1

    def set_wdir(self, htc, wdir, **_):
        htc.wind.windfield_rotations = wdir, 0, 0

    def set_shear(self, htc, shear, **_):
        if isinstance(shear, str):
            shear = eval(shear)  # convert str to dict (if via excel)
        shear_format, shear_arg = shear['profile']
        i_shear_format = ['log', 'power'].index(shear_format.lower()) + 2
        htc.wind.shear_format = i_shear_format, shear_arg
        if shear['type'] == 'NWP':
            return
        elif shear['type'] == 'EWS':
            phi = {'++': 0, '+-': 90, '--': 180, '-+': 270}[shear['sign']]
            htc.wind.iec_gust = 'ews', shear['A'], phi, self.time_start, shear['T']
        else:
            raise NotImplementedError(shear['type'])

    def set_ti(self, htc, ti, **_):
        htc.wind.tint = ti

    def set_seed(self, htc, seed, **kwargs):
        if seed is None or seed == "":
            htc.wind.turb_format = 0
        elif isinstance(seed, int):
            L, Gamma, nx, nyz = self.turbulence_defaults

            htc.add_mann_turbulence(L, 1, Gamma, seed, no_grid_points=(nx, nyz, nyz),
                                    box_dimension=(kwargs['simulation_time'] * kwargs['V_hub'],
                                                   self.diameter, self.diameter))
        else:
            raise NotImplementedError(seed)

    def set_Gust(self, htc, Gust, **kwargs):
        if str(Gust).lower() == 'nan':
            return
        if isinstance(Gust, str):
            Gust = eval(Gust)
        if Gust['type'] == 'ECD':
            V_cg, theta_cg, T = [Gust[k] for k in ['V_cg', 'theta_cg', 'T']]
            htc.wind.iec_gust = 'ECD', V_cg, theta_cg, self.time_start, T
        else:
            raise NotImplementedError(Gust)

    def set_Fault(self, htc, Fault, **kwargs):
        if str(Fault).lower() == 'nan':
            return
        if isinstance(Fault, str):
            Fault = eval(Fault)
        if Fault['type'] == 'GridLoss':
            T = Fault['T']
            self.set_gridloss_time(htc, self.time_start + T)
        else:
            raise NotImplementedError(Fault)

    def set_simulation_time(self, htc, simulation_time, **_):
        htc.set_time(self.time_start, simulation_time + self.time_start)

    def set_gridloss_time(self, htc, t):
        gen_servo = htc.dll.get_subsection_by_name('generator_servo', 'name')
        assert gen_servo.init.constant__7.comments == "Time for grid loss [s]"
        gen_servo.init.constant__7 = 7, t


if __name__ == '__main__':
    dlb = DTU_IEC64100_1_Ref_DLB(iec_wt_class='1A', Vin=4, Vout=26, Vr=10, D=180, z_hub=90)
    path = os.path.dirname(test_files.__file__) + '/simulation_setup/DTU10MWRef6.0/'
    writer = HAWC2_IEC_DLC_Writer(path + 'htc/DTU_10MW_RWT.htc', 180)
    p = writer.from_pandas(dlb['DLC14'])
    print(p.contents)

    p.write_all(out_dir='tmp')
