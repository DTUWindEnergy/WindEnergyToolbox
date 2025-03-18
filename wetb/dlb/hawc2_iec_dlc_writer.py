import os
import warnings
from wetb.hawc2.hawc2_input_writer import HAWC2InputWriter
from wetb.hawc2.tests import test_files
from wetb.dlb.iec61400_1 import DTU_IEC61400_1_Ref_DLB

"""
TODO: delete wind ramp / replace wind section
TODO: set default turb_format = 0
"""


class HAWC2_IEC_DLC_Writer(HAWC2InputWriter):
    def __init__(self, base_htc_file, diameter,
                 time_start=100,  # Minimum 5s cf. IEC61400-1(2005), section 7.5
                 turbulence_defaults=(33.6, 3.9, 8192, 64)  # L, gamma, n_x, n_yz):
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
            htc.wind.iec_gust = 'ecd', V_cg, theta_cg, self.time_start, T
        elif Gust['type'] == 'EDC':
            phi, T = [Gust[k] for k in ['phi', 'T']]
            htc.wind.iec_gust = 'edc', 0, phi, self.time_start, T
        elif Gust['type'] == 'EOG':
            V_gust, T = [Gust[k] for k in ['V_gust', 'T']]
            htc.wind.iec_gust = 'eog', V_gust, 0, self.time_start, T
        else:
            raise NotImplementedError(Gust)

    def set_Fault(self, htc, Fault, **kwargs):
        if str(Fault).lower() == 'nan':
            return
        if isinstance(Fault, str):
            Fault = eval(Fault)
        if Fault['type'] == 'GridLoss':
            generator_servo = Fault['generator_servo']
            T = Fault['T']
            self.set_gridloss_time(htc, generator_servo, self.time_start + T)
        elif Fault['type'] == 'StuckBlade':
            pitch_servo = Fault['pitch_servo']
            T = Fault['T']
            self.set_stuckblade(htc, pitch_servo, T)
        elif Fault['type'] == 'PitchRunaway':
            pitch_servo = Fault['pitch_servo']
            T = Fault['T']
            self.set_pitchrunaway(htc, pitch_servo, self.time_start + T)
        else:
            raise NotImplementedError(Fault)
            
    def set_Operation(self, htc, Operation, **kwargs):
        if str(Operation).lower() == 'nan':
            return
        if isinstance(Operation, str):
            Operation = eval(Operation)
        if Operation['type'] == 'StartUp':
            controller = Operation['controller']
            T = Operation['T']
            self.set_startup_time(htc, controller, self.time_start + T)
        elif Operation['type'] == 'ShutDown':
            controller = Operation['controller']
            T = Operation['T']
            self.set_shutdown_time(htc, controller, self.time_start + T, 1)
        elif Operation['type'] == 'EmergencyShutDown':
            controller = Operation['controller']
            T = Operation['T']
            self.set_shutdown_time(htc, controller, self.time_start + T, 2)
        elif Operation['type'] == 'Parked':
            controller = Operation['controller']
            self.set_parked(htc, controller, self.time_start + kwargs['simulation_time'])
        elif Operation['type'] == 'RotorLocked':
            controller = Operation['controller']
            self.set_parked(htc, controller, self.time_start + kwargs['simulation_time'])
            shaft = Operation['shaft']
            shaft_constraint = Operation['shaft_constraint']
            azimuth = Operation['Azi']
            self.set_rotor_locked(htc, shaft, shaft_constraint, azimuth)
        else:
            raise NotImplementedError(Operation)

    def set_simulation_time(self, htc, simulation_time, **_):
        htc.set_time(self.time_start, simulation_time + self.time_start)

    def set_gridloss_time(self, htc, generator_servo, t):
        generator_servo = htc.dll.get_subsection_by_name(generator_servo, 'name')
        if 'time for grid loss' not in generator_servo.init.constant__7.comments.lower():
            warnings.warn('Assuming constant 7 in generator_servo DLL is time for grid loss!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "time for grid loss" in constant 7 comment.')
        generator_servo.init.constant__7 = 7, t
        
    def set_stuckblade(self, htc, pitch_servo, t):
        pitch_servo = htc.dll.get_subsection_by_name(pitch_servo, 'name')
        if 'time for stuck blade' not in pitch_servo.init.constant__9.comments.lower():
            warnings.warn('Assuming constant 9 in pitch_servo DLL is time for stuck blade!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "time for stuck blade" in constant 9 comment.')
        pitch_servo.init.constant__9 = 9, t
        if 'angle of stuck blade' not in pitch_servo.init.constant__10.comments.lower():
            warnings.warn('Assuming constant 10 in pitch_servo DLL is angle of stuck blade!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "angle of stuck blade" in constant 10 comment.')
        pitch_servo.init.constant__10 = 10, 0
        
    def set_pitchrunaway(self, htc, pitch_servo, t):
        pitch_servo = htc.dll.get_subsection_by_name(pitch_servo, 'name')
        if 'time for pitch runaway' not in pitch_servo.init.constant__8.comments.lower():
            warnings.warn('Assuming constant 8 in pitch_servo DLL is time for pitch runaway!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "time for stuck blade" in constant 8 comment.')
        pitch_servo.init.constant__8 = 8, t
        
    def set_startup_time(self, htc, controller, t):
        controller = htc.dll.get_subsection_by_name(controller, 'name')
        if 'cut-in time' not in controller.init.constant__24.comments.lower():
            warnings.warn('Assuming constant 24 in controller DLL is cut-in time!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "cut-in time" in constant 24 comment.')
        controller.init.constant__24 = 24, t
        
    def set_shutdown_time(self, htc, controller, t, stop_type):
        controller = htc.dll.get_subsection_by_name(controller, 'name')
        if 'shut-down time' not in controller.init.constant__26.comments.lower():
            warnings.warn('Assuming constant 26 in controller DLL is shut-down time!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "shut-down time" in constant 26 comment.')
        controller.init.constant__26 = 26, t
        if 'stop type' not in controller.init.constant__28.comments.lower():
            warnings.warn('Assuming constant 28 in controller DLL is stop type!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "stop type" in constant 28 comment.')
        controller.init.constant__28 = 28, stop_type
        
    def set_parked(self, htc, controller, simulation_time):
        controller = htc.dll.get_subsection_by_name(controller, 'name')
        if 'cut-in time' not in controller.init.constant__24.comments.lower():
            warnings.warn('Assuming constant 24 in controller DLL is cut-in time!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "cut-in time" in constant 24 comment.')
        controller.init.constant__24 = 24, simulation_time + 1
        htc.aero.induction_method = 0
        
    def set_rotor_locked(self, htc, shaft, shaft_constraint, azimuth):
        for constraint in htc.new_htc_structure.constraint:
            try:
                if shaft_constraint in str(constraint.name):
                    constraint.name_ = 'bearing3'
                    constraint.add_line('omegas', [0])
                    break
            except:
                continue
        for orientation in htc.new_htc_structure.orientation:
            try:
                if shaft in str(orientation.mbdy2):
                    command = 'mbdy2_eulerang'
                    for line in orientation:
                        if 'mbdy2_eulerang' in line.name_:
                            command = line.name_
                    orientation[command].values[2] = azimuth                   
                    break
            except:
                continue
                  

if __name__ == '__main__':
    dlb = DTU_IEC61400_1_Ref_DLB(iec_wt_class='1A', Vin=4, Vout=26, Vr=10, D=180, z_hub=90)
    path = os.path.dirname(test_files.__file__) + '/simulation_setup/DTU10MWRef6.0/'
    writer = HAWC2_IEC_DLC_Writer(path + 'htc/DTU_10MW_RWT.htc', 180)
    p = writer.from_pandas(dlb['DLC14'])
    print(p.contents)

    p.write_all(out_dir='tmp')
