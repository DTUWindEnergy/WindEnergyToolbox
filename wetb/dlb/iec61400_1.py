
import os
import numpy as np
import pandas as pd
import itertools


"""IEC refers to IEC61400-1(2005)

Questions:
dlc1.4 (hansen: 100s, no turb), why not NTM, NWP 600s and 6 seeds
seed numbering and reuse???
"""


class DLC():

    def __init__(self, Description, Turb, Shear, Gust, Fault=None, variables={}):
        self.Description = Description
        self.Turb = Turb
        self.Shear = Shear
        self.Gust = Gust
        self.Fault = Fault
        self.variables = variables
        self.variables.update({k.lower(): v for k, v in variables.items()})
        turb_class = self.iec_wt_class[1].lower()
        assert turb_class in 'abc'
        self.I_ref = {'a': .16, 'b': .14, 'c': .12}[turb_class]  # IEC61400-1(2005) table 1
        wind_class = int(self.iec_wt_class[0])
        assert 1 <= wind_class <= 3
        self.V_ref = {1: 50, 2: 42.5, 3: 37.5}[wind_class]
        if variables["seed"]:
            self.rng = np.random.default_rng(seed=variables["seed"])

    @classmethod
    def getattr(cls, name):
        try:
            return getattr(cls, name)
        except AttributeError as e:
            d = {k.lower(): k for k in dir(cls)}
            if name.lower() in d:
                return getattr(cls, d[name.lower()])
            else:
                raise e

    def __getattr__(self, name):
        try:
            return self.__getattribute__(name)
        except AttributeError as e:
            if name in self.variables:
                return self.variables[name]
            elif name.lower() in self.variables:
                return self.variables[name.lower()]
            else:
                raise e

    def get_lst(self, x):
        if isinstance(x, pd.Series):
            x = x.iloc[0]
        if ":" in str(x):
            start, step, stop = [float(eval(v, globals(), self.variables)) for v in x.split(":")]
            return list(np.arange(start, stop + step, step))
        else:
            return [float(eval(v, globals(), self.variables)) for v in str(x).replace("/", ",").split(",")]

    @property
    def wsp_lst(self):
        return sorted(self.get_lst(self.WSP))

    @property
    def wdir_lst(self):
        return self.get_lst(self.Wdir)

    def case_arg_lst_product(self, **kwargs):
        case_arg_lst = []
        for dict_lst in itertools.product(
                *[m(self, **kwargs) for m in [DLC.WspWdir, self.Turb, self.Shear, self.Gust, self.Fault, self.Operation] if m is not None]):
            ids = {k: v for d in dict_lst for k, v in d.items() if '_id' in k}
            d = {k: v for d in dict_lst for k, v in d.items() if '_id' not in k}
            name = [self.Name, 'wsp%02d' % d['V_hub'], "wdir%03d" % (d['wdir'] % 360)]
            if 'seed_id' in ids:
                name.append("s%04d" % d['seed'])
            if 'ews_id' in ids:
                name.append("ews%s" % d['shear']['sign'])
            if 'edc_id' in ids:
                name.append(ids['edc_id'])
            if 'T_id' in ids:
                name.append(ids['T_id'])
            if 'Azi_id' in ids:
                name.append(ids['Azi_id'])
            d['Name'] = "_".join(name)
            case_arg_lst.append(d)
        return case_arg_lst

    def to_pandas(self, **extra_kwargs):
        case_dict_lst = []

        for V_hub in self.wsp_lst:
            for wdir in self.wdir_lst:
                default_kwargs = {'Folder': self.Name,
                                  'simulation_time': self.Time,
                                  'V_hub': V_hub,
                                  'wdir': wdir,
                                  }
                for case_args in self.case_arg_lst_product(**default_kwargs, **extra_kwargs):
                    case_args.update({k: v for k, v in default_kwargs.items() if k not in case_args})
                    case_dict_lst.append(case_args)
        cols = ['Name', 'Folder', 'V_hub', 'wdir', 'simulation_time']
        cols += [k for k in case_args.keys() if k not in cols]
        return pd.DataFrame(case_dict_lst, columns=cols)

    # ===============================================================================
    # General
    # ===============================================================================
    def WspWdir(self, V_hub, wdir, **_):
        return [{'V_hub': V_hub, 'wdir': wdir}]

    def NONE(self, **_):
        return [{}]

    def NaN(self, **_):
        return [{}]

    # ===============================================================================
    # Turbulence models
    # ===============================================================================
    def NoTurb(self, **_):
        return [{'seed': None}]

    def Turb_0p11(self, **_):
        s0 = 1001
        ti = 0.11
        if self.seed:
            return [{'seed_id': 's%04d' % (s),
                    'ti': ti,
                    'seed': s} for s in self.rng.integers(low=0, high=10000, size=int(self.Seeds))]
        else:
            return [{'seed_id': 's%04d' % (s),
                    'ti': ti,
                    'seed': s} for s in range(s0, s0 + int(self.Seeds))]

    def NTM(self, V_hub, **_):
        # Normal turbulence model IEC section 6.3.1.3
        s0 = int((V_hub - self.Vin) // self.Vstep * 100 + 1001)
        ti = (self.I_ref * (0.75 * V_hub + 5.6)) / V_hub  # IEC (11)
        if self.seed:
            return [{'seed_id': 's%04d' % (s),
                    'ti': ti,
                    'seed': s} for s in self.rng.integers(low=0, high=10000, size=int(self.Seeds))]
        else:
            return [{'seed_id': 's%04d' % (s),
                    'ti': ti,
                    'seed': s} for s in range(s0, s0 + int(self.Seeds))]


    def ETM(self, V_hub, **_):
        # Extreme Turbulence model
        # IEC (9)
        V_ave = .2 * self.V_ref

        # IEC (19)
        c = 2
        ti = c * self.I_ref * (0.072 * (V_ave / c + 3) * (V_hub / c - 4) + 10) / V_hub

        s0 = int((V_hub - self.Vin) // self.Vstep * 100 + 1001)

        if self.seed:
            return [{'seed_id': 's%04d' % (s),
                    'ti': ti,
                    'seed': s} for s in self.rng.integers(low=0, high=10000, size=int(self.Seeds))]
        else:
            return [{'seed_id': 's%04d' % (s),
                    'ti': ti,
                    'seed': s} for s in range(s0, s0 + int(self.Seeds))]

    # ===============================================================================
    # Shear profiles
    # ===============================================================================

    def NWP(self, **_):
        # The normal wind profile model IEC section 6.3.1.2
        return [{'shear': {'type': 'NWP', 'profile': ('power', .2)}}]
    
    def NWP_0p11(self, **_):
        # The normal wind profile model IEC section 6.3.1.2 for 50-year extreme wind conditions
        return [{'shear': {'type': 'NWP', 'profile': ('power', .11)}}]

    def EWS(self, V_hub, **_):
        # Extreme wind shear, IEC section 6.3.2.6
        beta = 6.4
        T = 12
        sigma_1 = self.I_ref * (0.75 * V_hub + 5.6)  # IEC (11)

        D = self.D
        lambda_1 = 42 if self.z_hub < 60 else .7 * self.z_hub
        A = (2.5 + 0.2 * beta * sigma_1 * (D / lambda_1)**0.25) / D  # IEC (26) & (27)

        return [{'shear': {'type': 'EWS', 'profile': ('power', .2), 'A': A, 'T': T, 'sign': ews_sign},
                 'ews_id': ews_sign}
                for ews_sign in ['++', '+-', '-+', '--']]

    # ===========================================================================
    # Gusts
    # ===========================================================================

    def ECD(self, V_hub, **_):
        # Extreme coherrent gust with direction change IEC 6.3.2.5
        # IEC section 6.3.2.5
        # IEC (22)
        V_cg = 15  # magnitude of gust
        # IEC (24)
        theta_cg = (720 / V_hub, 180)[V_hub < 4]  # direction change
        T = 10  # rise time
        return [{'Gust': {'type': 'ECD', 'V_cg': V_cg, 'theta_cg': theta_cg, 'T': T}}]
    
    def EDC(self, V_hub, **_):
        # Extreme direction change
        phi = 4*(np.rad2deg(np.arctan(self.I_ref*(0.75*V_hub + 5.6)/V_hub/(1 + 0.1*self.D/self.scale_parameter))))
        T = 10 # Duration
        return [{'Gust': {'type': 'EDC', 'phi': sign*phi, 'T': T}, 'edc_id': {-1: '-', 1: '+'}[sign]} for sign in [-1, 1]]
    
    def EOG(self, V_hub, **_):
        # Extreme operation gust
        V_gust = min(1.35*(0.8*1.4*self.V_ref - V_hub), 3.3*self.I_ref*(0.75*V_hub + 5.6)/(1 + 0.1*self.D/self.scale_parameter))  # magnitude of gust
        T = 10.5  # duration
        return [{'Gust': {'type': 'EOG', 'V_gust': V_gust, 'T': T}}]

    # ===============================================================================
    # Faults
    # ===============================================================================

    def StuckBlade(self, **_):
        return [{'Fault': {'type': 'StuckBlade', 'pitch_servo': self.pitch_servo, 'T': 0.1, 'pitch': 0}}]
    
    def PitchRunaway(self, **_):
        return [{'Fault': {'type': 'PitchRunaway', 'pitch_servo': self.pitch_servo, 'T': 10}}]

    def GridLoss10(self, **_):
        return [{'Fault': {'type': 'GridLoss', 'generator_servo': self.generator_servo, 'T': 10}}]
    
    def GridLoss3times(self, **_):
        gridloss_times = [2.5, 4, 5.25]
        return [{'Fault': {'type': 'GridLoss', 'generator_servo': self.generator_servo, 'T': T}, 'T_id': 't' + str(gridloss_times.index(T))} for T in gridloss_times]
    
    # ===============================================================================
    # Operation
    # ===============================================================================
    
    @staticmethod
    def Operation(self, **_):
        if 'start-up at 4 different times' in self.Description.lower():
            startup_times = [0.1, 2.5, 4, 5.25]
            return [{'Operation': {'type': 'StartUp', 'controller': self.controller, 'T': T}, 'T_id': 't' + str(startup_times.index(T))} for T in startup_times]
        if 'start-up at 2 different times' in self.Description.lower():
            startup_times = [-0.1, 5]
            return [{'Operation': {'type': 'StartUp', 'controller': self.controller, 'T': T}, 'T_id': 't' + str(startup_times.index(T))} for T in startup_times]
        if 'start-up' in self.Description.lower():
            return [{'Operation': {'type': 'StartUp', 'controller': self.controller, 'T': 0}}]
        if 'emergency shut-down' in self.Description.lower():
            return [{'Operation': {'type': 'EmergencyShutDown', 'controller': self.controller, 'T': 0}}]
        if 'shut-down at 6 different times' in self.Description.lower():
            shutdown_times = [0.1, 2.5, 4, 5, 8, 10]
            return [{'Operation': {'type': 'ShutDown', 'controller': self.controller, 'T': T}, 'T_id': 't' + str(shutdown_times.index(T))} for T in shutdown_times]
        if 'shut-down' in self.Description.lower():
            return [{'Operation': {'type': 'ShutDown', 'controller': self.controller, 'T': 0}}]
        if 'parked' in self.Description.lower():
            return [{'Operation': {'type': 'Parked', 'controller': self.controller}}]
        if 'rotor locked at 4 different azimuth angles' in self.Description.lower():
            azimuth_angles = [0, 30, 60, 90]
            return [{'Operation': {'type': 'RotorLocked', 'controller': self.controller, 'shaft': self.shaft, 'shaft_constraint': self.shaft_constraint, 'Azi': azi},
                     'Azi_id': 'azi' + f"{azi:03}"} for azi in azimuth_angles]
        if 'rotor locked' in self.Description.lower():
            return [{'Operation': {'type': 'RotorLocked', 'controller': self.controller, 'shaft': self.shaft, 'shaft_constraint': self.shaft_constraint, 'Azi': self.best_azimuth}}]
        else:
            return [{}]
        

class DLB():
    
    def __init__(self, dlc_definitions, variables):
        cols = ['Name', 'Description', 'WSP', 'Wdir', 'Turb', 'Seeds', 'Shear', 'Gust', 'Fault', 'Time']
        self.dlcs = pd.DataFrame(dlc_definitions, columns=cols, index=[dlc['Name'] for dlc in dlc_definitions])

        var_name_desc = [('iec_wt_class', 'IEC wind turbine class, e.g. 1A'),
                         ('tiref', 'Reference turbulence intensity'),
                         ('Vin', 'Cut-in wind speed'),
                         ('Vout', 'Cut-out wind speed'),
                         ('Vr', 'Rated wind speed'),
                         ('Vref', 'Reference wind speed'),
                         ('V1', '1-year recurrence period wind speed'),
                         ('Ve50', 'Extreme 50-year recurrence period wind speed'),
                         ('Ve1', 'Extreme 1-year recurrence period wind speed'),
                         ('Vmaint', 'Maximum wind speed for maintenance'),
                         ('Vstep', 'Wind speed distribution step'),
                         ('D', 'Rotor diameter'),
                         ('z_hub', 'Hub height'),
                         ('scale_parameter', 'Scale parameter'),
                         ('controller', 'Filename of controller DLL'),
                         ('generator_servo', 'Filename of generator servo DLL'),
                         ('pitch_servo', 'Filename of pitch servo DLL'),
                         ('best_azimuth', 'Best blade azimuth for maintenance'),
                         ('shaft', 'Name of shaft body'),
                         ('shaft_constraint', 'Name of constraint between tower and shaft'),
                         ("seed", "Seed to initialize the RNG for turbulence seed generation")
                         ]
        self.variables = pd.DataFrame([{'Name': n, 'Value': variables[n], 'Description': d}
                                       for n, d in var_name_desc], columns=['Name', 'Value', 'Description'],
                                      index=[n for (n, d) in var_name_desc])

    @property
    def dlb(self):
        if not hasattr(self, '_dlb'):
            self.generate_DLB()
        return self._dlb

    def keys(self):
        return [n for n in self.dlcs['Name'] if not str(n).lower().strip() in ['', 'none', 'nan']]

    def to_excel(self, filename):
        if os.path.dirname(filename) != "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        writer = pd.ExcelWriter(filename)
        self.dlcs.to_excel(writer, 'DLC', index=False)
        self.variables.to_excel(writer, 'Variables', index=False)
        writer.close()

    @staticmethod
    def from_excel(filename):
        df_vars = pd.read_excel(filename, sheet_name='Variables',
                                index_col='Name')
        df_vars.fillna('', inplace=True)
        variables = {name: value for name, value in zip(df_vars.index, df_vars.Value.values)}

        dlb_def = pd.read_excel(filename, 'DLC')
        dlb_def.columns = [c.strip() for c in dlb_def.columns]
        return DLB([row for _, row in dlb_def.iterrows() if row['Name'] is not np.nan], variables)

    def __getitem__(self, key):
        return self.dlb[key]

    def _make_dlc(self, dlc_name):
        dlc_definition = self.dlcs.loc[dlc_name]
        kwargs = {k: DLC.getattr(str(dlc_definition[k]))
                  for k in ['Turb', 'Shear', 'Gust', 'Fault']}
        kwargs['Description'] = dlc_definition['Description']
        variables = {v['Name']: v['Value'] for _, v in self.variables.iterrows()}
        variables.update(dlc_definition)
        dlc = DLC(variables=variables, **kwargs)
        extra_kwargs = {}
        extra_kwargs['Vref'] = variables['Vref']
        extra_kwargs['tiref'] = variables['tiref']
        extra_kwargs['D'] = variables['D']
        extra_kwargs['scale_parameter'] = variables['scale_parameter']
        df = dlc.to_pandas(**extra_kwargs)
        name = dlc_definition['Name']
        df.insert(0, 'DLC', name)
        return df

    def generate_DLB(self):
        self._dlb = {dlc: self._make_dlc(dlc) for dlc in self.keys()}
        return self._dlb

    def to_pandas(self):
        return pd.concat(self.dlb.values(), sort=False)

    def cases_to_excel(self, filename):
        if os.path.dirname(filename) != "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        writer = pd.ExcelWriter(filename)
        for k in self.keys():
            self[k].to_excel(writer, k, index=False)
        writer.close()


class DTU_IEC61400_1_Ref_DLB(DLB):
    
    def __init__(self, iec_wt_class, Vin, Vout, Vr, Vmaint, D, z_hub,
                 controller, generator_servo, pitch_servo, best_azimuth,
                 Vstep=2, seed=None, shaft='shaft', shaft_constraint='shaft_rot'):
        """
        NOTE!!!!!!!!!!!
        SEVERAL DLCS ARE MISSING
        """
        
        Name, Description, WSP, Wdir, Time = 'Name', 'Description', 'WSP', 'Wdir', 'Time'
        Turb, Seeds, Shear, Gust, Fault = 'Turb', 'Seeds', 'Shear', 'Gust', 'Fault'

        dlc_definitions = [
            {Name: 'DLC12',
             Description: 'Normal production',
             WSP: 'Vin:Vstep:Vout',
             Wdir: '-10/0/10',               
             Turb: 'NTM',
             Seeds: 6,
             Shear: 'NWP',
             Gust: None,
             Fault: None,
             Time: 600},
            
            {Name: 'DLC13',
             Description: 'Normal production with high turbulence',
             WSP: 'Vin:Vstep:Vout',
             Wdir: '-10/0/10',                
             Turb: 'ETM',
             Seeds: 6,
             Shear: 'NWP',
             Gust: None,
             Fault: None,
             Time: 600},
            
            {Name: 'DLC14',
             Description: 'Normal production with gust and direction change',
             WSP: 'Vr/Vr+2/Vr-2',
             Wdir: 0,
             Turb: 'NoTurb',
             Seeds: None,
             Shear: 'NWP',
             Gust: 'ECD',
             Fault: None,
             Time: 100},
            
            {Name: 'DLC15',
             Description: 'Normal production with extreme wind shear',
             WSP: 'Vin:Vstep:Vout',
             Wdir: 0,
             Turb: 'NoTurb',
             Seeds: None,
             Shear: 'EWS',
             Gust: None,
             Fault: None,
             Time: 100},
            
            {Name: 'DLC21',
             Description: 'Loss of electical network',
             WSP: 'Vin:Vstep:Vout',
             Wdir: '-10/0/10',
             Turb: 'NTM',
             Seeds: 4,
             Shear: 'NWP',
             Gust: None,
             Fault: 'GridLoss10',
             Time: 100},
            
            {Name: 'DLC22b',
             Description: 'One blade stuck at minimum pitch angle',
             WSP: 'Vin:Vstep:Vout',
             Wdir: 0,
             Turb: 'NTM',
             Seeds: 12,
             Shear: 'NWP',
             Gust: None,
             Fault: 'StuckBlade',
             Time: 100},
            
            {Name: 'DLC22p',
             Description: 'Pitch runaway',
             WSP: 'Vr:Vstep:Vout',
             Wdir: 0,
             Turb: 'NTM',
             Seeds: 12,
             Shear: 'NWP',
             Gust: None,
             Fault: 'PitchRunaway',
             Time: 100},
            
            {Name: 'DLC22y',
             Description: 'Abnormal yaw error',
             WSP: 'Vin:Vstep:Vout',
             Wdir: '15:15:345',
             Turb: 'NTM',
             Seeds: 1,
             Shear: 'NWP',
             Gust: None,
             Fault: None,
             Time: 600},
            
            {Name: 'DLC23',
              Description: 'Loss of electical network with extreme operating gust',
              WSP: 'Vr-2/Vr+2/Vout',
              Wdir: 0,
              Turb: 'NoTurb',
              Seeds: None,
              Shear: 'NWP',
              Gust: 'EOG',
              Fault: 'GridLoss3times',
              Time: 100},
            
            {Name: 'DLC24',
             Description: 'Normal production with large yaw error',
             WSP: 'Vin:Vstep:Vout',
             Wdir: '-20/20',
             Turb: 'NTM',
             Seeds: 3,
             Shear: 'NWP',
             Gust: None,
             Fault: None,
             Time: 600},
            
            {Name: 'DLC31',
             Description: 'Start-up',
             WSP: 'Vin/Vr/Vout',
             Wdir: 0,
             Turb: 'NoTurb',
             Seeds: None,
             Shear: 'NWP',
             Gust: None,
             Fault: None,
             Time: 100},
            
            {Name: 'DLC32',
              Description: 'Start-up at 4 different times with extreme operating gust',
              WSP: 'Vin/Vr-2/Vr+2/Vout',
              Wdir: 0,
              Turb: 'NoTurb',
              Seeds: None,
              Shear: 'NWP',
              Gust: 'EOG',
              Fault: None,
              Time: 100},
            
            {Name: 'DLC33',
              Description: 'Start-up at 2 different times with extreme wind direction change',
              WSP: 'Vin/Vr-2/Vr+2/Vout',
              Wdir: 0,
              Turb: 'NoTurb',
              Seeds: None,
              Shear: 'NWP',
              Gust: 'EDC',
              Fault: None,
              Time: 100},
            
            {Name: 'DLC41',
             Description: 'Shut-down',
             WSP: 'Vin/Vr/Vout',
             Wdir: 0,
             Turb: 'NoTurb',
             Seeds: None,
             Shear: 'NWP',
             Gust: None,
             Fault: None,
             Time: 100},
            
            {Name: 'DLC42',
              Description: 'Shut-down at 6 different times with extreme operating gust',
              WSP: 'Vr-2/Vr+2/Vout',
              Wdir: 0,
              Turb: 'NoTurb',
              Seeds: None,
              Shear: 'NWP',
              Gust: 'EOG',
              Fault: None,
              Time: 100},
            
            {Name: 'DLC51',
              Description: 'Emergency shut-down',
              WSP: 'Vr-2/Vr+2/Vout',
              Wdir: 0,
              Turb: 'NTM',
              Seeds: 12,
              Shear: 'NWP',
              Gust: None,
              Fault: None,
              Time: 100},
            
            {Name: 'DLC61',
             Description: 'Parked with 50-year wind',
             WSP: 'Vref',
             Wdir: '-8/8',
             Turb: 'Turb_0p11',
             Seeds: 6,
             Shear: 'NWP_0p11',
             Gust: None,
             Fault: None,
             Time: 600},
            
            {Name: 'DLC62',
             Description: 'Parked with 50-year wind without grid connection',
             WSP: 'Vref',
             Wdir: '0:15:345',
             Turb: 'Turb_0p11',
             Seeds: 1,
             Shear: 'NWP_0p11',
             Gust: None,
             Fault: None,
             Time: 600},
            
            {Name: 'DLC63',
             Description: 'Parked with 1-year wind with large yaw error',
             WSP: 'V1',
             Wdir: '-20/20',
             Turb: 'Turb_0p11',
             Seeds: 6,
             Shear: 'NWP_0p11',
             Gust: None,
             Fault: None,
             Time: 600},
            
            {Name: 'DLC64',
             Description: 'Parked',
             WSP: 'Vin:Vstep:0.7*Vref',
             Wdir: '-8/8',
             Turb: 'NTM',
             Seeds: 6,
             Shear: 'NWP',
             Gust: None,
             Fault: None,
             Time: 600},
            
            {Name: 'DLC71',
             Description: 'Rotor locked at 4 different azimuth angles and extreme yaw',
             WSP: 'V1',
             Wdir: '0:15:345',
             Turb: 'Turb_0p11',
             Seeds: 1,
             Shear: 'NWP_0p11',
             Gust: None,
             Fault: None,
             Time: 600},
            
            {Name: 'DLC81',
             Description: 'Rotor locked for maintenance',
             WSP: 'Vmaint',
             Wdir: '-8/8',
             Turb: 'NTM',
             Seeds: 6,
             Shear: 'NWP',
             Gust: None,
             Fault: None,
             Time: 600},
        ]
        
        Vref = {1: 50, 2: 42.5, 3: 37.5}[int(iec_wt_class[0])]
        tiref = {'a': 0.16, 'b': 0.14, 'c': 0.12}[iec_wt_class[1].lower()]
        V1 = 0.8*Vref
        Ve50 = 1.4*Vref
        Ve1 = 0.8*Ve50
        scale_parameter = 42 # TO-DO: define it as function of hub height
        
        variables = {'iec_wt_class': iec_wt_class,
                     'Vref': Vref,
                     'tiref': tiref,
                     'V1': V1,
                     'Ve50': Ve50,
                     'Ve1': Ve1,
                     'Vin': Vin,
                     'Vout': Vout,
                     'Vr': Vr,
                     'Vmaint': Vmaint,
                     'Vstep': Vstep,
                     'D': D,
                     'z_hub': z_hub,
                     'scale_parameter': scale_parameter,
                     'controller': controller,
                     'generator_servo': generator_servo,
                     'pitch_servo': pitch_servo,
                     'best_azimuth': best_azimuth,
                     'shaft': shaft,
                     'shaft_constraint': shaft_constraint}
        if seed:
            variables["seed"] = int(seed)
        else:
            variables["seed"] = seed
        DLB.__init__(self, dlc_definitions, variables)

class DTU_IEC61400_1_Ref_DLB_custom(DLB):
    def __init__(self, iec_wt_class, Vin, Vout, Vr, D, z_hub, seed=None, time=10):
        """
        NOTE!!!!!!!!!!!
        SEVERAL DLCS ARE MISSING
        """
        Vstep = 2
        Name, Description, WSP, Wdir, Time = 'Name', 'Description', 'WSP', 'Wdir', 'Time'
        Turb, Seeds, Shear, Gust, Fault = 'Turb', 'Seeds', 'Shear', 'Gust', 'Fault'

        dlc_definitions = [
            {Name: 'DLC12', Description: 'Normal production', WSP: 'Vin:2:Vout', Wdir: '-10/0/10',
                Turb: 'NTM', Seeds: 6, Shear: 'NWP', Gust: None, Fault: None, Time: time},
            {Name: 'DLC13', Description: 'Normal production with high turbulence', WSP: 'Vin:2:Vout', Wdir: '-10/0/10',
                Turb: 'ETM', Seeds: 6, Shear: 'NWP', Gust: None, Fault: None, Time: time},
            {Name: 'DLC14', Description: 'Normal production with gust and direction change', WSP: 'Vr/Vr+2/Vr-2', Wdir: 0,
                Turb: 'NoTurb', Seeds: None, Shear: 'NWP', Gust: 'ECD', Fault: None, Time: time},
            {Name: 'DLC15', Description: 'Normal production with extreme wind shear', WSP: 'Vin:2:Vout', Wdir: 0,
                Turb: 'NoTurb', Seeds: None, Shear: 'EWS', Gust: None, Fault: None, Time: time},
            {Name: 'DLC21', Description: 'Loss of electical network', WSP: 'Vin:2:Vout', Wdir: '-10/0/10',
                Turb: 'NTM', Seeds: 4, Shear: 'NWP', Gust: None, Fault: 'GridLoss10', Time: time},
            {Name: 'DLC22y', Description: 'Abnormal yaw error', WSP: 'Vin:2:Vout', Wdir: '15:15:345',
                Turb: 'NTM', Seeds: 1, Shear: 'NWP', Gust: None, Fault: None, Time: time}
        ]
        variables = {'iec_wt_class': iec_wt_class, 'Vin': Vin,
                     'Vout': Vout, 'Vr': Vr, 'D': D, 'z_hub': z_hub, 'Vstep': Vstep}
        if seed:
            variables["seed"] = int(seed)
        else:
            variables["seed"] = seed
        DLB.__init__(self, dlc_definitions, variables)

def main():
    if __name__ == '__main__':

        dlb = DTU_IEC61400_1_Ref_DLB(iec_wt_class='1A',
                                     Vin=4, Vout=25, Vr=8,  # cut-in, cut_out and rated wind speed
                                     D=180, z_hub=110)  # diameter and hub height
        print(dlb.dlcs)
        print(dlb.variables)

        # save dlb definition to excel
        dlb.to_excel('overview.xlsx')
        # you can now modify definitions in overview.xlsx in Excel and
        # load the modified dlb definition back into python
        dlb = DLB.from_excel('overview.xlsx')

        print(dlb['DLC14'])

        # save dlc14 as Excel spreadsheet
        dlb['DLC14'].to_excel('dlc14.xlsx')
        # you can no modify cases in dlc14.xlsx in Excel

        # save all cases to excel
        dlb.cases_to_excel('cases.xlsx')

        # Save generate hawc2 input files
        from wetb.hawc2.tests import test_files
        from wetb.dlb.hawc2_iec_dlc_writer import HAWC2_IEC_DLC_Writer
        path = os.path.dirname(test_files.__file__) + '/simulation_setup/DTU10MWRef6.0/'
        writer = HAWC2_IEC_DLC_Writer(path + 'htc/DTU_10MW_RWT.htc', diameter=180)

        # load all cases
        writer.from_pandas(dlb)
        # load DLC14 only
        writer.from_pandas(dlb['DLC14'])
        # load modified DLC14.xlsx
        writer.from_excel('dlc14.xlsx')

        writer.write_all('tmp')


main()
