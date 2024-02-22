import os
import warnings
from wetb.hawc2.hawc2_input_writer import HAWC2InputWriter
from wetb.hawc2.tests import test_files
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
        htc.wind.wind_ramp_factor = 0, self.time_start*0.75, 0.1 / V_hub, 1

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
        
    def WriteWaveInput(self,folder,WaveType,Hs,Tp,seed):
        WT = {'Irregular': 1,'Regular': 0}[WaveType]
        if WT == 1:
            name = [f'./wavedata/wave_type{WT}_hs{int(round(Hs,2)*100)}_tp{int(round(Tp,2)*100)}',"_s%05d" % (seed),'.inp']
            Wavename = "".join(name)
            # Wavename = f'./wavedata/{WaveType}_{'Hs%02d' % d[Hs]}_{Tp}_{WaveSeed}.inp'
        elif WT == 0:
            name = [f'./wavedata/wave_type{WT}_hs{int(round(Hs,2)*100)}_tp{int(round(Tp,2)*100)}','.inp']
            Wavename = "".join(name)
        os.makedirs(os.path.dirname(folder+Wavename), exist_ok=True)
        with open(folder+Wavename,'w+') as fid:
            fid.write(f'''begin wkin_input ;
  wavetype {WT} ; 0=regular, 1=irregular, 2=deterministic
  wdepth 30.0 ;
  ;
  begin reg_airy ;
    stretching 0; 0=none, 1=wheeler
    wave {Hs} {Tp}; Hs,T
  end;
  ;
  begin ireg_airy ;
    stretching 0; 0=none, 1=wheeler
    coef 200 {seed} ; number of coefficients, seed
    spectrum 2; 1=jonswap, 2=Pierson Moscowitz
    ;jonswap 1.102 8.515 1.0 ; Jonswap: Hs, Tp, gamma
    pm {Hs} {Tp} ; Pierson Moscowitz: Hs, Tp, gamma
    spreading 0 2; Spreading model [0=off 1=on], Spreading parameter [pos. integer min 1]
  end;
  ;
end wkin_input;
exit ;''')
        return Wavename

    def set_Wave(self, htc, Wave, seed, **_):        
        if Wave['WaveType'] == 'Irregular' or Wave['WaveType'] == 'Regular':
            # write wave input file
            # Wave['seed'] = seed
            WaveInputName =self.WriteWaveInput(folder=htc.modelpath,**Wave,seed=seed)
            hydro = htc.add_section('hydro')
            water_properties = hydro.add_section('water_properties')
            htc.hydro.water_properties.water_kinematics_dll = 'wkin_dll.dll', WaveInputName
        elif Wave['WaveType'] == 'nan':
            return
        else:
            raise NotImplementedError(Wave['WaveType'])
        
    def set_SeaLevel(self, htc, SeaLevel, **_):
        hydro = htc.add_section('hydro')
        water_properties = hydro.add_section('water_properties')
        htc.hydro.water_properties.mwl = SeaLevel

    def set_Wavedir(self, htc, Wavedir, **_):
        hydro = htc.add_section('hydro')
        water_properties = hydro.add_section('water_properties')
        htc.hydro.water_properties.wave_direction = Wavedir

    def set_Fault(self, htc, Fault, **kwargs):
        if str(Fault).lower() == 'nan':
            return
        if isinstance(Fault, str):
            Fault = eval(Fault)
        if Fault['type'] == 'GridLoss':
            T = Fault['T']
            self.set_gridloss_time(htc, self.time_start + T)
        elif Fault['type'] == 'Start-up':
            htc.dll.get_subsection_by_name('dtu_we_controller').init.constant__24[1] = Fault['T']
        elif Fault['type'] == 'Shut-down':
            htc.dll.get_subsection_by_name('dtu_we_controller').init.constant__26[1] = Fault['T']
        elif Fault['type'] == 'Idle':
            htc.dll.get_subsection_by_name('dtu_we_controller').init.constant__26[1] = 0.1
        elif Fault['type'] == 'Locked_rotor':
            Warning("Locked rotor fault is currently only implemented as a hardcoded 'continue_in_file' for the IEA-15MW-RWT")
            d = Fault['angle']
            htc.new_htc_structure.orientation.continue_in_file = "../IEA-15-240-RWT/IEA_15MW_RWT_WTG_orientation_shaftfix_%02ddeg.htc" % d
            htc.new_htc_structure.constraint.continue_in_file  = "../IEA-15-240-RWT/IEA_15MW_RWT_WTG_constraint_shaftfix.htc"
        else:
            raise NotImplementedError(Fault)

    def set_simulation_time(self, htc, simulation_time,**_):
        htc.set_time(self.time_start, simulation_time + self.time_start)

    def set_gridloss_time(self, htc, t):
        gen_servo = htc.dll.get_subsection_by_name('generator_servo', 'name')
        if 'time for grid loss' not in gen_servo.init.constant__7.comments.lower():
            warnings.warn('Assuming constant 7 in generator_servo DLL is time for grid loss!'
                          ' Please verify your htc file is correct. Disable warning by '
                          + 'placing "time for grid loss" in constant 7 comment.')
        gen_servo.init.constant__7 = 7, t


if __name__ == '__main__':
    dlb = DTU_IEC64100_1_Ref_DLB(iec_wt_class='1A', Vin=4, Vout=26, Vr=10, D=180, z_hub=90)
    path = os.path.dirname(test_files.__file__) + '/simulation_setup/DTU10MWRef6.0/'
    writer = HAWC2_IEC_DLC_Writer(path + 'htc/DTU_10MW_RWT.htc', 180)
    p = writer.from_pandas(dlb['DLC14'])
    print(p.contents)

    p.write_all(out_dir='tmp')
