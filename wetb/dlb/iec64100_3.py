import os
import numpy as np
import pandas as pd
import itertools

"""IEC refers to IEC61400-3(2005)

Questions:
dlc1.4 (hansen: 100s, no turb), why not NTM, NWP 600s and 6 seeds
seed numbering and reuse???
"""

class DLC():
    #remember wavetable
    def __init__(self, Description, Turb, Shear, Gust, Waves, Fault=None, variables={}):
        self.Description = Description
        self.Turb = Turb
        self.Shear = Shear
        self.Gust = Gust
        self.Waves = Waves
        self.WaveTable = pd.read_excel(variables['WaveTable'])
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
        if "NWLR" in str(x):
            return [self.LAT, self.MSL, self.HAT]
        if "MSL" in str(x):
            return [self.MSL]
        if "LockedRotor" in str(x):
            return [0,30,60,90]
        else:
            return [float(eval(v, globals(), self.variables)) for v in str(x).replace("/", ",").split(",")]

    @property
    def wsp_lst(self):
        return sorted(self.get_lst(self.WSP))

    @property
    def wdir_lst(self):
        return self.get_lst(self.Wdir)
    
    @property
    def Wavedir_lst(self):
        return self.get_lst(self.Wavedir)
    
    @property
    def SeaLevel_lst(self):
        return self.get_lst(self.sealevel)
    
    @property
    def Fault_lst(self):
        if self.Fault.__name__ == 'LockedRotor':
            return self.get_lst(self.Fault)
        else:
            return [1]

    
    # @property
    # def seastate_lst(self):
    #     return self.get_lst(self.Wavedir)
    
    # @property
    # def SeaLevel_lst(self):
    #     return [self.LAT, self.MSL, self.HAT]

    def case_arg_lst_product(self, **kwargs):
        case_arg_lst = []
        for dict_lst in itertools.product(
                *[m(self, **kwargs) for m in [DLC.WspWdir, self.Turb, self.Shear, self.Gust, DLC.Wavedirec, self.Waves, DLC.NWLR_MSL, self.Fault] if m is not None]):
            ids = {k: v for d in dict_lst for k, v in d.items() if '_id' in k}
            d = {k: v for d in dict_lst for k, v in d.items() if '_id' not in k}
            name = [self.Name, 'wsp%02d' % d['V_hub'], "wdir%03d" % (d['wdir'] % 360), 'wavedir%03d' % (d['RelativeWavedir']), d['SeaLevel_str']]
            if 'seed_id' in ids:
                name.append("s%05d" % d['seed'])
            if 'ews_id' in ids:
                name.append("ews%s" % d['shear']['sign'])
            if self.Fault.__name__ == 'StartUp':
                name.append('startup')
            if self.Fault.__name__ == 'ShutDown':
                name.append('shutdown')
            if self.Fault.__name__ == 'Idle':
                name.append('idle')
            if self.Fault.__name__ == 'LockedRotor':
                name.append('lockedrotor%02d' % d['Fault']['angle'])

            d['Name'] = "_".join(name)
            case_arg_lst.append(d)

        return case_arg_lst

    def to_pandas(self):
        case_dict_lst = []

        for V_hub in self.wsp_lst:
            for wdir in self.wdir_lst:
                for Wavedir in self.Wavedir_lst:
                    for SeaLevel in self.SeaLevel_lst:
                        for Fault in self.Fault_lst:
                            default_kwargs = {'Folder': self.Name,
                                            'simulation_time': self.Time,
                                            'V_hub': V_hub,
                                            'wdir': wdir,
                                            'Wavedir': Wavedir,
                                            'SeaLevel': SeaLevel,
                                            }
                            if Fault != 1:
                                default_kwargs["Fault"] = Fault

                            for case_args in self.case_arg_lst_product(**default_kwargs):
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

    def NTM(self, V_hub, wdir, Wavedir, **_):
        
        # Waveseed and windseed 
        Wavedir_step = {'-22.5': 1,'-10.0': 2, '0.0': 3, '10.0': 4, '22.5': 5}[str(Wavedir)]
        wdir_step    = {'-20.0': 1,'-10.0': 2, '-8.0': 3, '0.0': 4,  '8.0': 5, '10.0': 6, '20.0': 7}[str(wdir)]

        # Normal turbulence model IEC section 6.3.1.3
        s0 = int(wdir_step*1000 + Wavedir_step*10000 + (V_hub - self.Vin) // self.Vstep * 10 + 1)
        ti = (self.I_ref * (0.75 * V_hub + 5.6)) / V_hub  # IEC (11)
        
        if self.seed:
            return [{'seed_id': 's%05d' % (s),
                    'ti': ti,
                    'seed': s} for s in self.rng.integers(low=0, high=100000, size=int(self.Seeds))]
        else:
            return [{'seed_id': 's%05d' % (s),
                    'ti': ti,
                    'seed': s} for s in range(s0, s0 + int(self.Seeds))]


    def ETM(self, V_hub, **_):
        # Extreme Turbulence model
        # IEC (9)
        V_ave = .2 * self.V_ref

        # IEC (19)
        c = 2
        ti = c * self.I_ref * (0.072 * (V_ave / c + 3) * (V_hub / c - 4) + 10) / V_hub

        s0 = int((V_hub - self.Vin) // self.Vstep * 100 + 10001)
        
        if self.seed:
            return [{'seed_id': 's%05d' % (s),
                    'ti': ti,
                    'seed': s} for s in self.rng.integers(low=0, high=100000, size=int(self.Seeds))]
        else:
            return [{'seed_id': 's%05d' % (s),
                    'ti': ti,
                    'seed': s} for s in range(s0, s0 + int(self.Seeds))]  

    # ===============================================================================
    # Shear profiles
    # ===============================================================================

    def NWP(self, **_):
        # The normal wind profile model IEC section 6.3.1.2
        return [{'shear': {'type': 'NWP', 'profile': ('power', .08)}}]

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

    # ===============================================================================
    # Wave and current parameters
    # ===============================================================================

    def Wavedirec(self, wdir, Wavedir, **_):
        return [{'Wavedir': wdir+Wavedir, 'RelativeWavedir': Wavedir}]
    
    def Irregular(self, V_hub, **_):
        # df1 = pd.read_excel(Wavefile)
        # hs,Tp = f(V_hub), wvwseed = f(v_hub,wdir,wavedir)
        [Hs, Tp] = self.WaveTable.loc[self.WaveTable['Wsp'] == V_hub][['Hm0','Tp']].values[0]
        return [{'Wave': {'WaveType': 'Irregular', 'Hs': Hs, 'Tp': Tp}}]# ,'WaveSeed': Wavedir_step*3000+wdir_step*300+int(V_hub)}]
    
    def Regular(self, V_hub, **_):
        # hs,Tp = f(V_hub), wvwseed = f(v_hub,wdir,wavedir)
        [Hs, Tp] = self.WaveTable.loc[self.WaveTable['Wsp'] == V_hub][['Hs','Tp']].values[0]
        return [{'Wave': {'WaveType': 'Regular', 'Hs': Hs, 'Tp': Tp}}] #, 'WaveSeed': 1}}]
    
    def NWLR_MSL(self,SeaLevel, **_):
        if SeaLevel < 0:
            id = 'LAT'
        elif SeaLevel == 0:
            id = 'MSL'
        elif SeaLevel > 0:
            id = 'HAT'

        return [{'SeaLevel': SeaLevel, 'SeaLevel_str': id}]
    

    # ===============================================================================
    # Faults
    # ===============================================================================

    def GridLoss10(self, **_):
        return [{'Fault': {'type': 'GridLoss', 'T': 10}}]
    
    def StartUp(self, **_):
        return [{'Fault': {'type': 'Start-up', 'T': 300}}]
    
    def ShutDown(self, **_):
        return [{'Fault': {'type': 'Shut-down', 'T': 300}}]
    
    def Idle(self, **_):
        return [{'Fault': {'type': 'Idle'}}]
    
    def LockedRotor(self, Fault, **_):
        return [{'Fault': {'type': 'Locked_rotor', 'angle': Fault}}]
    

class DLB():
    def __init__(self, dlc_definitions, variables):
        cols = ['Name', 'Description', 'WSP', 'Wdir', 'Turb', 'Seeds', 'Shear', 'Gust', 'Wavedir' ,'Waves', 'SeaLevel', 'Fault', 'Time']
        self.dlcs = pd.DataFrame(dlc_definitions, columns=cols, index=[dlc['Name'] for dlc in dlc_definitions])

        var_name_desc = [('Vin', 'Cut-in wind speed'),
                         ('Vout', 'Cut-out wind speed'),
                         ('Vr', 'Rated wind speed'),
                         ('VrefX07', '0.7 x Reference wind speed'),
                         ('MSL', 'Mean sea level'),
                         ('LAT', 'Lowest astronomical tide'),
                         ('HAT', 'Highest astronomical tide'),
                         ('WaveTable', 'Dataframe with Hs and Tp'),
                         ('D', 'Rotor diameter'),
                         ('z_hub', 'Hub height'),
                         ('Vstep', 'Wind speed distribution step'),
                         ('iec_wt_class', 'IEC wind turbine class, e.g. 1A'),
                         ("seed", "Seed to initialize the RNG for turbulence seed generation")
                         ]
        self.variables = pd.DataFrame([{'Name': n, 'Value': variables[n], 'Description': d}
                                       for n, d in var_name_desc], columns=['Name', 'Value', 'Description'],
                                      index=[n for (n, d) in var_name_desc])
        

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
        dlc_definition = self.dlcs.loc[key]
        kwargs = {k: DLC.getattr(str(dlc_definition[k]))
                  for k in ['Turb', 'Shear', 'Gust', 'Waves', 'Fault']}
        kwargs['Description'] = dlc_definition['Description']
        variables = {v['Name']: v['Value'] for _, v in self.variables.iterrows()}
        variables.update(dlc_definition)
        dlc = DLC(variables=variables, **kwargs)
        df = dlc.to_pandas()
        name = dlc_definition['Name']
        df.insert(0, 'DLC', name)
        return df

    def to_pandas(self):
        return pd.concat([self[dlc] for dlc in self.keys()], sort=False)

    def cases_to_excel(self, filename):
        if os.path.dirname(filename) != "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        writer = pd.ExcelWriter(filename)
        for k in self.keys():
            self[k].to_excel(writer, k, index=False)
        writer.close()

class DTU_IEC64100_3_Ref_DLB(DLB):
    def __init__(self, iec_wt_class, Vin, Vout, Vr, VrefX07, MSL, LAT, HAT, WaveTable, D, z_hub, seed=None):
        """
        NOTE!!!!!!!!!!!
        SEVERAL DLCS ARE MISSING
        """
        Vstep = 2
        Name, Description, WSP, Wdir, Time = 'Name', 'Description', 'WSP', 'Wdir', 'Time'
        Turb, Seeds, Shear, Gust, Wavedir, Waves, SeaLevel, Fault = 'Turb', 'Seeds', 'Shear', 'Gust', 'Wavedir', 'Waves', 'SeaLevel', 'Fault'

        dlc_definitions = [
            {Name: 'DLC12', Description: 'Normal production', WSP: 'Vin:2:Vout', Wdir: '-10/0/10',
                Turb: 'NTM', Seeds: 6, Shear: 'NWP', Gust: None, Wavedir: '-22.5/0/22.5', Waves: 'Irregular', SeaLevel: 'NWLR', Fault: None, Time: 600}, #OK
            {Name: 'DLC24', Description: 'Power production, large yaw error', WSP: 'Vin:2:Vout', Wdir: '-20/20', 
                Turb: 'NTM', Seeds: 6, Shear: 'NWP', Gust: None, Wavedir: '0', Waves: 'Irregular', SeaLevel: 'MSL', Fault: None, Time: 600}, #OK
            {Name: 'DLC31', Description: 'Start-up, normal wind profile' , WSP: 'Vin/Vr/Vout', Wdir: '0',
                Turb: 'NTM', Seeds: 6, Shear: 'NWP', Gust: None, Wavedir: '0', Waves: 'Irregular', SeaLevel: 'MSL', Fault: 'StartUp', Time: 600}, #OK
            {Name: 'DLC41', Description: 'Shut-down, normal wind profile' , WSP: 'Vin/Vr/Vout', Wdir: '0',
                Turb: 'NTM', Seeds: 6, Shear: 'NWP', Gust: None, Wavedir: '0', Waves: 'Irregular', SeaLevel: 'MSL', Fault: 'ShutDown', Time: 600}, #OK
            {Name: 'DLC64', Description: 'Parked, idling rotor' , WSP: 'Vin:2:VrefX07', Wdir: '-8/8',
                Turb: 'NTM', Seeds: 6, Shear: 'NWP', Gust: None, Wavedir: '0', Waves: 'Irregular', SeaLevel: 'NWLR', Fault: 'Idle', Time: 600}, #OK
            {Name: 'DLC72', Description: 'Rotor locked, normal wind conditions' , WSP: 'Vin:2:Vout', Wdir: '-10/0/10',
                Turb: 'NTM', Seeds: 6, Shear: 'NWP', Gust: None, Wavedir: '0', Waves: 'Irregular', SeaLevel: 'NWLR', Fault: 'LockedRotor', Time: 600} # OK
        ]
        variables = {'iec_wt_class': iec_wt_class, 'Vin': Vin,
                     'Vout': Vout, 'Vr': Vr, 'VrefX07': VrefX07, 
                     'MSL': MSL, 'LAT': LAT, 'HAT': HAT,
                     'WaveTable': WaveTable,
                     'D': D, 'z_hub': z_hub, 'Vstep': Vstep}
        if seed:
            variables["seed"] = int(seed)
        else:
            variables["seed"] = seed
        
        DLB.__init__(self, dlc_definitions, variables)


class DTU_IEC64100_3_Ref_DLB_custom(DLB):
    def __init__(self, iec_wt_class, Vin, Vout, Vr, VrefX07, MSL, LAT, HAT, WaveTable, D, z_hub, seed=None, time=10):
        """
        NOTE!!!!!!!!!!!
        SEVERAL DLCS ARE MISSING
        """
        Vstep = 2
        Name, Description, WSP, Wdir, Time = 'Name', 'Description', 'WSP', 'Wdir', 'Time'
        Turb, Seeds, Shear, Gust, Wavedir, Waves, SeaLevel, Fault = 'Turb', 'Seeds', 'Shear', 'Gust', 'Wavedir', 'Waves', 'SeaLevel', 'Fault'

        dlc_definitions = [
            {Name: 'DLC12', Description: 'Normal production', WSP: 'Vin:2:Vout', Wdir: '-10/0/10',
                Turb: 'NTM', Seeds: 6, Shear: 'NWP', Gust: None, Wavedir: '-22.5/0/22.5', Waves: 'Irregular', SeaLevel: 'NWLR', Fault: None, Time: time}, #OK
            {Name: 'DLC24', Description: 'Power production, large yaw error', WSP: 'Vin:2:Vout', Wdir: '-20/20', 
                Turb: 'NTM', Seeds: 6, Shear: 'NWP', Gust: None, Wavedir: '0', Waves: 'Irregular', SeaLevel: 'MSL', Fault: None, Time: time}, #OK
            {Name: 'DLC31', Description: 'Start-up, normal wind profile' , WSP: 'Vin/Vr/Vout', Wdir: '0',
                Turb: 'NTM', Seeds: 6, Shear: 'NWP', Gust: None, Wavedir: '0', Waves: 'Irregular', SeaLevel: 'MSL', Fault: 'StartUp', Time: time}, #OK
            {Name: 'DLC41', Description: 'Shut-down, normal wind profile' , WSP: 'Vin/Vr/Vout', Wdir: '0',
                Turb: 'NTM', Seeds: 6, Shear: 'NWP', Gust: None, Wavedir: '0', Waves: 'Irregular', SeaLevel: 'MSL', Fault: 'ShutDown', Time: time}, #OK
            {Name: 'DLC64', Description: 'Parked, idling rotor' , WSP: 'Vin:2:VrefX07', Wdir: '-8/8',
                Turb: 'NTM', Seeds: 6, Shear: 'NWP', Gust: None, Wavedir: '0', Waves: 'Irregular', SeaLevel: 'NWLR', Fault: 'Idle', Time: time}, #OK
            {Name: 'DLC72', Description: 'Rotor locked, normal wind conditions' , WSP: 'Vin:2:Vout', Wdir: '-10/0/10',
                Turb: 'NTM', Seeds: 6, Shear: 'NWP', Gust: None, Wavedir: '0', Waves: 'Irregular', SeaLevel: 'NWLR', Fault: 'LockedRotor', Time: time} # OK
        ]
        variables = {'iec_wt_class': iec_wt_class, 'Vin': Vin,
                     'Vout': Vout, 'Vr': Vr, 'VrefX07': VrefX07, 
                     'MSL': MSL, 'LAT': LAT, 'HAT': HAT,
                     'WaveTable': WaveTable,
                     'D': D, 'z_hub': z_hub, 'Vstep': Vstep}
        if seed:
            variables["seed"] = int(seed)
        else:
            variables["seed"] = seed
        DLB.__init__(self, dlc_definitions, variables)


if __name__ == '__main__':
    
    Wavepath =  'C:/Users/rsod/Downloads/hawc2_iec_dlc_writer/'
    Wavefile = Wavepath + 'WaveTable.xlsx'

    dlb = DTU_IEC64100_3_Ref_DLB(iec_wt_class='1B',
                                    Vin=4, Vout=24, Vr=10.69, VrefX07=34,  # cut-in, cut_out and rated wind speed
                                    WaveTable= Wavefile,
                                    MSL=0, LAT=-0.30, HAT=0.35, # sea level parameters
                                    D=240, z_hub=150)     # diameter and hub height
    print(dlb.dlcs)
    print(dlb.variables)

    # save dlb definition to excel
    dlb.to_excel('./overview.xlsx')
    # you can now modify definitions in overview.xlsx in Excel and
    # load the modified dlb definition back into python
    dlb = DLB.from_excel('./overview.xlsx')

    # print(dlb['DLC72'])

    # # save dlc14 as Excel spreadsheet
    # dlb['DLC12'].to_excel('dlc12.xlsx')
    # # you can no modify cases in dlc14.xlsx in Excel

    # save all cases to excel
    dlb.cases_to_excel('cases.xlsx')

    # Save generate hawc2 input files
    from wetb.hawc2.tests import test_files
    from wetb.dlb.hawc2_iec_dlc_writer import HAWC2_IEC_DLC_Writer

    path =  r'C:\Users\rsod\source\repos\IEA-15-240-RWT\HAWC2\IEA-15-240-RWT-Monopile/'
    writer = HAWC2_IEC_DLC_Writer(path + 'htc/IEA_15MW_RWT_Monopile.htc', diameter=240)

    # load all cases
    writer.from_pandas(dlb)
    # # load DLC14 only
    # writer.from_pandas(dlb['DLC12'])
    # # load modified DLC14.xlsx
    # writer.from_excel('dlc12.xlsx')

    writer.write_all(path+'tmp')
