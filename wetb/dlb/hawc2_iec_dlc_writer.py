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
    def __init__(self, base_htc_file,
                 diameter=300, # For turb box size, should be passed unless instantiated from DLB
                 time_start=100,  # Minimum 5s cf. IEC61400-1(2005), section 7.5
                 turbulence_defaults=(33.6, 3.9, 8192, 64),  # L, gamma, n_x, n_yz)
                 controller='dtu_we_controller',
                 generator_servo='generator_servo',
                 pitch_servo='servo_with_limits',
                 constant_cutin=24,
                 constant_cutout=26,
                 constant_shutdown_type=28,
                 stop_type={'Normal': 1, 'Emergency': 2},
                 constant_gridloss_time=7,
                 constant_stuckblade_time=9,
                 constant_stuckblade_angle=10,
                 constant_pitchrunaway_time=8,
                 shaft_mbdy='shaft',
                 shaft_constraint='shaft_rot'):
        HAWC2InputWriter.__init__(self, base_htc_file,
                                  diameter=diameter,
                                  time_start=time_start,
                                  turbulence_defaults=turbulence_defaults,
                                  controller=controller,
                                  generator_servo=generator_servo,
                                  pitch_servo=pitch_servo,
                                  constant_cutin=constant_cutin,
                                  constant_cutout=constant_cutout,
                                  constant_shutdown_type=constant_shutdown_type,
                                  stop_type=stop_type,
                                  constant_gridloss_time=constant_gridloss_time,
                                  constant_stuckblade_time=constant_stuckblade_time,
                                  constant_stuckblade_angle=constant_stuckblade_angle,
                                  constant_pitchrunaway_time=constant_pitchrunaway_time,
                                  shaft_mbdy=shaft_mbdy,
                                  shaft_constraint=shaft_constraint)

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
            if hasattr(self, 'lambda_1'):
                L = 0.8 * self.lambda_1
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
            T = Fault['T']
            self.set_gridloss_time(htc, self.generator_servo, self.constant_gridloss_time, self.time_start + T)
        elif Fault['type'] == 'StuckBlade':
            T = Fault['T']
            pitch = Fault['pitch']
            self.set_stuckblade(htc, self.pitch_servo, self.constant_stuckblade_time, self.constant_stuckblade_angle, T, pitch)
        elif Fault['type'] == 'PitchRunaway':
            T = Fault['T']
            self.set_pitchrunaway(htc, self.pitch_servo, self.constant_pitchrunaway_time, self.time_start + T)
        else:
            raise NotImplementedError(Fault)
            
    def set_Operation(self, htc, Operation, **kwargs):
        if str(Operation).lower() == 'nan':
            return
        if isinstance(Operation, str):
            Operation = eval(Operation)
        if Operation['type'] == 'StartUp':
            T = Operation['T']
            self.set_startup_time(htc, self.controller, self.constant_cutin, self.time_start + T)
        elif Operation['type'] == 'ShutDown':
            T = Operation['T']
            self.set_shutdown_time(htc, self.controller, self.constant_cutout, self.constant_shutdown_type, self.time_start + T, self.stop_type['Normal'])
        elif Operation['type'] == 'EmergencyShutDown':
            T = Operation['T']
            self.set_shutdown_time(htc, self.controller, self.constant_cutout, self.constant_shutdown_type, self.time_start + T, self.stop_type['Emergency'])
        elif Operation['type'] == 'Parked':
            self.set_parked(htc, self.controller, self.constant_cutin, self.time_start + kwargs['simulation_time'])
        elif Operation['type'] == 'RotorLocked':
            self.set_parked(htc, self.controller, self.constant_cutin, self.time_start + kwargs['simulation_time'])
            azimuth = Operation['Azi']
            self.set_rotor_locked(htc, self.shaft_mbdy, self.shaft_constraint, azimuth)
        else:
            raise NotImplementedError(Operation)

    def set_simulation_time(self, htc, simulation_time, **_):
        htc.set_time(self.time_start, simulation_time + self.time_start)

    def set_gridloss_time(self, htc, generator_servo, constant_gridloss_time, t):
        generator_servo = htc.dll.get_subsection_by_name(generator_servo, 'name')
        if 'time for grid loss' not in getattr(generator_servo.init, f'constant__{constant_gridloss_time}').comments.lower():
            warnings.warn(f'Assuming constant {constant_gridloss_time} in generator_servo DLL is time for grid loss!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "time for grid loss" in constant {constant_gridloss_time} comment.')
        setattr(generator_servo.init, f'constant__{constant_gridloss_time}', [constant_gridloss_time, t])
        
    def set_stuckblade(self, htc, pitch_servo, constant_stuckblade_time, constant_stuckblade_angle, t, pitch):
        pitch_servo = htc.dll.get_subsection_by_name(pitch_servo, 'name')
        if 'time for stuck blade' not in getattr(pitch_servo.init, f'constant__{constant_stuckblade_time}').comments.lower():
            warnings.warn(f'Assuming constant {constant_stuckblade_time} in pitch_servo DLL is time for stuck blade!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "time for stuck blade" in constant {constant_stuckblade_time} comment.')
        setattr(pitch_servo.init, f'constant__{constant_stuckblade_time}', [constant_stuckblade_time, t])
        if 'angle of stuck blade' not in getattr(pitch_servo.init, f'constant__{constant_stuckblade_angle}').comments.lower():
            warnings.warn(f'Assuming constant {constant_stuckblade_angle} in pitch_servo DLL is angle of stuck blade!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "angle of stuck blade" in constant {constant_stuckblade_angle} comment.')
        setattr(pitch_servo.init, f'constant__{constant_stuckblade_angle}', [constant_stuckblade_angle, pitch])
        
    def set_pitchrunaway(self, htc, pitch_servo, constant_pitchrunaway_time, t):
        pitch_servo = htc.dll.get_subsection_by_name(pitch_servo, 'name')
        if 'time for pitch runaway' not in getattr(pitch_servo.init, f'constant__{constant_pitchrunaway_time}').comments.lower():
            warnings.warn(f'Assuming constant {constant_pitchrunaway_time} in pitch_servo DLL is time for pitch runaway!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "time for stuck blade" in constant {constant_pitchrunaway_time} comment.')
        setattr(pitch_servo.init, f'constant__{constant_pitchrunaway_time}', [constant_pitchrunaway_time, t])
        
    def set_startup_time(self, htc, controller, constant_cutin, t):
        controller = htc.dll.get_subsection_by_name(controller, 'name')
        if 'cut-in time' not in getattr(controller.init, f'constant__{constant_cutin}').comments.lower():
            warnings.warn(f'Assuming constant {constant_cutin} in controller DLL is cut-in time!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "cut-in time" in constant {constant_cutin} comment.')
        setattr(controller.init, f'constant__{constant_cutin}', [constant_cutin, t])
        
    def set_shutdown_time(self, htc, controller, constant_cutout, constant_shutdown_type, t, stop_type):
        controller = htc.dll.get_subsection_by_name(controller, 'name')
        if 'shut-down time' not in getattr(controller.init, f'constant__{constant_cutout}').comments.lower():
            warnings.warn(f'Assuming constant {constant_cutout} in controller DLL is shut-down time!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "shut-down time" in constant {constant_cutout} comment.')
        setattr(controller.init, f'constant__{constant_cutout}', [constant_cutout, t])
        if 'stop type' not in getattr(controller.init, f'constant__{constant_shutdown_type}').comments.lower():
            warnings.warn(f'Assuming constant {constant_shutdown_type} in controller DLL is stop type!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "stop type" in constant {constant_shutdown_type} comment.')
        setattr(controller.init, f'constant__{constant_shutdown_type}', [constant_shutdown_type, stop_type])
        
    def set_parked(self, htc, controller, constant_cutin, simulation_time):
        controller = htc.dll.get_subsection_by_name(controller, 'name')
        if 'cut-in time' not in getattr(controller.init, f'constant__{constant_cutin}').comments.lower():
            warnings.warn(f'Assuming constant {constant_cutin} in controller DLL is cut-in time!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "cut-in time" in constant {constant_cutin} comment.')
        setattr(controller.init, f'constant__{constant_cutin}', [constant_cutin, simulation_time + 1])
        htc.aero.induction_method = 0
        
    def set_rotor_locked(self, htc, shaft_mbdy, shaft_constraint, azimuth):
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
                if shaft_mbdy in str(orientation['mbdy2' if 'mbdy2' in orientation.keys() else 'body2']):
                    orientation.add_line('mbdy2_eulerang', [0, 0, azimuth], 'azimuth')
                    if 'mbdy2_ini_rotvec_d1' in orientation.keys():
                        orientation.contents.move_to_end('mbdy2_ini_rotvec_d1')
                    if 'body2_ini_rotvec_d1' in orientation.keys():
                        orientation.contents.move_to_end('body2_ini_rotvec_d1')
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
