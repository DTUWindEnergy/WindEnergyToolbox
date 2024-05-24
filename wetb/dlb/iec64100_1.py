
import os
import numpy as np
import pandas as pd
import itertools
from _collections import OrderedDict
from wetb.dlc.high_level import Weibull_IEC
import functools
import re
import h5py
import warnings


"""IEC refers to IEC61400-1(2005)

"""


class DLC():

    def __init__(self, DLC, Description, variables={}, Seeds=1, propability=None, **fields):
        self.Description = Description
        self.propability = propability
        self.name = DLC
        self.fields = fields
        self.variables = variables
        self.variables.update({k.lower(): v for k, v in variables.items()})
        self.seed = variables.get("seed", False)
        if self.seed:
            self.rng = np.random.default_rng(seed=self.seed)

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

    def get_field_value_lst(self, field):
        if isinstance(field, str) and hasattr(self, field):
            return getattr(self, field)()
        if isinstance(field, pd.Series):
            field = field.iloc[0]

        def fmt(v):
            for f in [float, int]:
                try:
                    v = f(v)
                except BaseException:
                    pass
            return v

        if ":" in str(field):
            start, step, stop = [fmt(eval(v, globals(), self.variables)) for v in field.split(":")]
            return list(np.arange(start, stop + step, step))
        else:
            try:
                return [fmt(eval(v, globals(), {field: field, 'nan': None, **self.variables}))
                        for v in str(field).replace("/", ",").split(",")]
            except BaseException:
                print

    def get_name_part(self, field, value):

        if field.lower() == 'wsp':
            return f'wsp{value:02d}'
        elif field.lower() == 'wdir':
            return f'wdir{value%360:03d}'
        if isinstance(value, dict) and 'name' in value:
            return value.pop('name')
        return f'{field}{value}'

    def case_arg_lst_product(self, **kwargs):
        case_arg_lst = []
        field_value_dict = {k: self.get_field_value_lst(v) for k, v in self.fields.items()
                            if not k.lower().endswith('_dist')}
        fields4name = {k: i for i, (k, v) in enumerate(field_value_dict.items()) if len(v) > 1}
        if self.fields.get('Fatigue', False):
            no_dist_prop = 1
            for n in fields4name:
                if f'{n}_dist' not in self.fields:
                    print(f"'{n}_dist' missing for multivalue '{n}' ({self.fields[n]}). Assuming uniform distribution")
                    no_dist_prop /= len(field_value_dict[n])
        #         assert f'{n}_dist' in self.fields, f"'{n}_dist' required for multivalue '{n}' ({self.fields[n]}) when Fatigue={self.fields['Fatigue']}"
        for dict_lst in itertools.product(*field_value_dict.values()):
            # *[m(self, **kwargs) for m in [DLC.WspWdir, self.Turb, self.Shear, self.Gust, self.Fault] if m is not None]):
            d = {f: v for f, v in zip(field_value_dict.keys(), dict_lst)}
            d.update(**kwargs)
            d.update({k: f(**d) for k, f in d.items() if callable(f)})

            d['Name'] = "_".join([self.name] + [self.get_name_part(n, d[n]) for n, i in fields4name.items()])
            # ids = {k: v for d in dict_lst for k, v in d.items() if '_id' in k}
            # d = {k: v for d in dict_lst for k, v in d.items() if '_id' not in k}
            # name = [self.DLC, 'wsp%02d' % d['V_hub'], "wdir%03d" % (d['wdir'] % 360)]
            #
            # if 'seed_id' in ids:
            #     name.append("s%04d" % d['seed'])
            # if 'ews_id' in ids:
            #     name.append("ews%s" % d['shear']['sign'])
            #d['Name'] = "_".join(name)

            if self.propability:
                n_lst, p = self.propability
                for n in n_lst:
                    p = p.get(d[n], p.get(str(d[n]).lower()))
                d['propability'] = p * no_dist_prop
            case_arg_lst.append(d)
        return case_arg_lst

    def to_pandas(self):
        default_kwargs = {'Folder': self.DLC,
                          'DLC': self.DLC,

                          }
        case_dict_lst = self.case_arg_lst_product(**default_kwargs)
        cols = ['DLC', 'Folder', 'Name']
        cols += [k for k in case_dict_lst[0].keys() if k not in cols]
        return pd.DataFrame(case_dict_lst, columns=cols)

    # ===============================================================================
    # General
    # ===============================================================================
    def WspWdir(self, WSP, Wdir, **_):
        return [{'V_hub': WSP, 'wdir': Wdir}]

    def NONE(self, **_):
        return [{}]

    def NaN(self, **_):
        return [{}]

    # ===============================================================================
    # Turbulence models
    # ===============================================================================
    def NoTurb(self, **_):
        return [{'seed': None}]

    def _TM(self, seed, WSP, ti_func, **_):

        # Normal turbulence model IEC section 6.3.1.3
        # ti = (self.Iref * (0.75 * WSP + 5.6)) / WSP  # IEC (11)
        if not self.seed:
            seed += int((WSP - self.Vin) // self.Vstep * 100 + 1001)
        return {'name': 's%04d' % (seed),
                'ti': ti_func(WSP),
                'seed': seed}

    def get_seed_lst(self):
        if self.seed:
            return self.rng.integers(low=0, high=10000, size=int(self.Seeds))
        else:
            return np.arange(int(self.Seeds))

    def NTM(self, **_):
        # Normal turbulence model IEC section 6.3.1.3

        def ti_NTM(WSP):
            # Normal turbulence model IEC section 6.3.1.3
            return (self.Iref * (0.75 * WSP + 5.6)) / WSP  # IEC (11)

        return [lambda seed=s, **kwargs:self._TM(seed, ti_func=ti_NTM, **kwargs) for s in self.get_seed_lst()]

    def ETM(self, **_):
        # Extreme Turbulence model

        def ti_ETM(WSP):
            # IEC (9)
            V_ave = .2 * self.Vref
            # IEC (19)
            c = 2
            return c * self.Iref * (0.072 * (V_ave / c + 3) * (WSP / c - 4) + 10) / WSP

        return [lambda seed=s, **kwargs:self._TM(seed, ti_func=ti_ETM, **kwargs) for s in self.get_seed_lst()]

    # ===============================================================================
    # Shear profiles
    # ===============================================================================

    def NWP(self, **_):
        # The normal wind profile model IEC section 6.3.1.2
        return [{'type': 'NWP', 'profile': ('power', .2)}]

    def EWS(self, **_):
        def EWS(ews_sign, WSP, **_):
            # Extreme wind shear, IEC section 6.3.2.6
            beta = 6.4
            T = 12
            sigma_1 = self.Iref * (0.75 * WSP + 5.6)  # IEC (11)

            D = self.D
            lambda_1 = 42 if self.z_hub < 60 else .7 * self.z_hub
            A = (2.5 + 0.2 * beta * sigma_1 * (D / lambda_1)**0.25) / D  # IEC (26) & (27)

            return {'type': 'EWS', 'profile': ('power', .2), 'A': A, 'T': T, 'sign': ews_sign, 'ews_id': ews_sign}
        return [lambda ews_sign=ews_sign, **kwargs: EWS(ews_sign, **kwargs) for ews_sign in ['++', '+-', '-+', '--']]

    # ===========================================================================
    # Gusts
    # ===========================================================================

    def ECD(self, **_):
        # Extreme coherrent gust with direction change IEC 6.3.2.5
        def ECD(WSP, **_):
            # IEC section 6.3.2.5
            # IEC (22)
            V_cg = 15  # magnitude of gust
            # IEC (24)
            theta_cg = (720 / WSP, 180)[WSP < 4]  # direction change
            T = 10  # rise time
            return {'type': 'ECD', 'V_cg': V_cg, 'theta_cg': theta_cg, 'T': T}
        return [ECD]

    # ===============================================================================
    # Faults
    # ===============================================================================

    def GridLoss10(self, **_):
        return [{'type': 'GridLoss', 'T': 10}]


class DLB():
    def __init__(self, dlc_definitions, variables):

        cols = list(dlc_definitions[0].keys())
        assert all([len(set(dlc_def.keys()) - set(cols)) == 0 for dlc_def in dlc_definitions]
                   ), "All fields must exists in the first dlc"
        self.dlcs = pd.DataFrame(dlc_definitions, columns=cols, index=[dlc['DLC'] for dlc in dlc_definitions])

        wind_class, turb_class = variables['iec_wt_class'].lower()
        wind_class = int(wind_class)
        assert turb_class in 'abc'
        assert 1 <= wind_class <= 3
        variables['Iref'] = {'a': .16, 'b': .14, 'c': .12}[turb_class]  # IEC61400-1(2005) table 1
        variables['Vref'] = {1: 50, 2: 42.5, 3: 37.5}[wind_class]
        var_name_desc = [('Vin', 'Cut-in wind speed'),
                         ('Vout', 'Cut-out wind speed'),
                         ('Vr', 'Rated wind speed'),
                         ('Vref', "Reference wind speed average over 10min (IEC61400-1(2005) table 1)"),
                         ('Iref', 'expected value of the turbulence intensity at 15m/s (IEC61400-1(2005) table 1)'),
                         ('D', 'Rotor diameter'),
                         ('z_hub', 'Hub height'),
                         ('Vstep', 'Wind speed distribution step'),
                         ('iec_wt_class', 'IEC wind turbine class, e.g. 1A'),
                         ('years', 'Years of operation in fatigue load calculation'),
                         ("seed", "Seed to initialize the RNG for turbulence seed generation")
                         ]
        self.variables = pd.DataFrame([{'Name': n, 'Value': variables[n], 'Description': d}
                                       for n, d in var_name_desc if n in variables],
                                      columns=['Name', 'Value', 'Description'],
                                      index=[n for (n, d) in var_name_desc if n in variables])

    @property
    def dlb(self):
        if not hasattr(self, '_dlb'):
            self.generate_DLB()
        return self._dlb

    def keys(self):
        return [n for n in self.dlcs['DLC'] if not str(n).lower().strip() in ['', 'none', 'nan']]

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
        return DLB([row for _, row in dlb_def.iterrows() if row['DLC'] is not np.nan], variables)

    def __getitem__(self, key):
        return self.dlb[key]

    def _make_dlc(self, dlc_name):
        dlc_definition = self.dlcs.loc[dlc_name]

        variables = {v['Name']: v['Value'] for _, v in self.variables.iterrows()}
        variables.update(dlc_definition)
        dlc = DLC(
            variables=variables,
            **dlc_definition.to_dict(),
            propability=self.get_case_propabilities(
                dlc_definition,
                variables))
        df = dlc.to_pandas()
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

    def get_case_propabilities(self, dlc_definition, variables):
        variables = {k.lower(): v for k, v in variables.items()}
        if str(variables.get('fatigue', 'nan')) == 'nan':
            return

        def get_propabilities(name, values, dist):
            values = str(values).lower().replace('/', ',')
            dist = str(dist).lower().replace('/', ',')
            if ":" in str(values):
                start, step, stop = [float(eval(v, globals(), variables)) for v in values.split(":")]
                values = np.arange(start, stop + step, step)
            else:
                try:
                    values = [(eval(v, globals(), variables)) for v in values.split(",")]
                except (SyntaxError, NameError):
                    try:
                        values = [(eval(v.lstrip('0'), globals(), variables))
                                  for v in values.split(",")]
                    except Exception:
                        values = values.split(",")

            if str(dist).lower() == "weibull" or str(dist).lower() == "rayleigh":
                dist = Weibull_IEC(variables['vref'], values)
            else:
                def fmt(v):
                    if str(v)[0] == '#':
                        return float(v[1:]) * variables['time'] / 3600 / (24 * 365)
                    else:
                        if v == "":
                            return 0
                        else:
                            return float(v) / 100
                dist = [fmt(v) for v in str(dist).split(",")]
            m = f"Number of {name}-values ({len(values)}) does not match number of {name}_dist-values({len(dist)}) in {variables['dlc']}"
            assert len(values) == len(dist), m
            return dict(zip(values, dist))

        # get dict: {'DLC': {'dlc12': 0.975}, 'WSP': {4.0: 0.11, 6.0: 0.14, ...}, 'Wdir': {-10: 0.25, 0: 0.5, ...}}
        propabilities = {n: get_propabilities(n, dlc_definition[n], dlc_definition[f'{n}_dist'])
                         for n in dlc_definition.keys() if f'{n}_dist' in dlc_definition.keys()}
        propability_fields = list(propabilities.keys())

        def get_prop(parent_propability, d_lst_index):
            if d_lst_index == len(propability_fields):
                return parent_propability
            else:
                return {value: get_prop(parent_propability * value_propability, d_lst_index + 1)
                        for value, value_propability in propabilities[propability_fields[d_lst_index]].items()}

        # return {'dlc12': {4.0: {-10: 39.15, 0: 78.31, 10: 39.15}, 6.0: {-10: 50.23,...},...},...}
        case_propability = 1 / variables['seeds']
        case_propability_dict = get_prop(case_propability, 0)
        return propability_fields, case_propability_dict

    def make_sensor_statistic_files(self, stat_file, sensors):
        import xarray as xr
        da = xr.load_dataarray(stat_file).sortby('filename')
        ext = os.path.splitext(da.filename[0].item())[1]
        cases = self['DLC12']
        re_pattern = re.compile(
            r"(?P<name>dlc(?P<dlc>\d\d.{0,2})_wsp(?P<wsp>\d\d)_wdir(?P<wdir>\d\d\d).*)" + ext,
            re.IGNORECASE)
        m = re_pattern.match(os.path.basename(da.filename[0].item()))
        if m is None:
            raise Exception(f"{re_pattern.pattern} does not match {da.filename[0].item()} (case insensitive)")

        names = list(m.groupdict().keys())
        units = [""] * len(names)
        descs = [""] * len(names)
        names += ['weight'] + da.stat.values.tolist()
        units += ['-'] + ['unit'] * len(da.stat)

        years = self.variables.loc['years'].Value
        life_time_sec = years * 365 * 24 * 3600
        descs += [f'No times this case occurs during {years} years'] + [''] * len(da.stat)
        data = list(zip(*[re_pattern.match(os.path.basename(f)).groups() for f in da.filename.values]))
        name_index_dict = {n: i for i, n in enumerate(data[0])}
        indexes = [name_index_dict[n] for n in cases.Name]

        # da.sel(filename=f"{cases.iloc[0].Folder}/{cases.iloc[0].Name}{ext}")

        folder = ''
        if folder not in ['', '.']:
            os.makedirs(folder, exist_ok=True)
        for _, sensor in sensors.iterrows():
            sensor_da = da.isel(sensor_name=np.array(eval(str(sensor.nr))) - 2)[indexes]
            weight = cases.propability.values * life_time_sec / cases.simulation_time.values
            sensor_data = data + [weight] + sensor_da.values.T.tolist()

            df = pd.DataFrame(zip(*sensor_data), columns=names)
            filename = os.path.join(folder, sensor.loc['name'] + ".h5")

            f = h5py.File(filename, "w")
            f.close()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df.to_hdf(filename, 'table')
            f = h5py.File(filename, "a")
            try:
                f.attrs["type"] = "PdapH5"
                f.attrs['name'] = sensor['name']
                f.attrs['description'] = sensor.description
                f.create_dataset("attribute_names", data=np.array([n.encode('utf-8') for n in names]))
                f.create_dataset("attribute_units", data=np.array([([u, sensor.unit][u == 'unit']).encode('utf-8')
                                                                   for u in units]))
                f.create_dataset("attribute_descriptions", data=np.array(["".encode('utf-8') for d in names]))
            finally:
                f.close()


class Sensors(pd.DataFrame):
    def __init__(self, data_frame):
        pd.DataFrame.__init__(self, data_frame)

    @staticmethod
    def from_excel(filename):
        def sensor_info(sensor_df, sensors=[]):
            if sensors != []:
                sensors = np.atleast_1d(sensors)
                empty_column = pd.DataFrame([""] * len(sensor_df.name))[0]
                return sensor_df[functools.reduce(
                    np.logical_or, [((sensor_df.get(f, empty_column).values != "") | (sensor_df.name == f)) for f in sensors])]
            else:
                return sensor_df

        # Sensors sheet
        sensor_df = pd.read_excel(filename, 'Sensors')
        # empty strings are now nans, convert back to empty strings
        sensor_df.fillna('', inplace=True)
        # force headers to lower case
        sensor_df.columns = [k.lower() for k in sensor_df.columns]

        for k in ['Name', 'Nr']:
            assert k.lower() in sensor_df.keys(), "Sensor sheet must have a '%s' column" % k
        sensor_df = sensor_df[sensor_df.name != ""]
        assert not any(sensor_df['name'].duplicated()), "Duplicate sensor names: %s" % ",".join(
            sensor_df['name'][sensor_df['name'].duplicated()].values)
        for k in ['description', 'unit', 'statistic', 'ultimate', 'fatigue', 'm',
                  'neql', 'extremeload', 'bearingdamage', 'mindistance', 'maxdistance']:
            if k not in sensor_df.keys():
                sensor_df[k] = ""
        for _, row in sensor_df[sensor_df['fatigue'] != ""].iterrows():
            msg = "Invalid m-value for %s (m='%s')" % (row['name'], row['m'])
            assert isinstance(row['m'], (int, float)), msg
            msg = "Invalid NeqL-value for %s (NeqL='%s')" % (row['name'], row['neql'])
            assert isinstance(row['neql'], (int, float)), msg
        for name, nrs in zip(sensor_info(sensor_df, "extremeload").name, sensor_info(sensor_df, "extremeload").nr):
            msg = "'Nr' for Extremeload-sensor '%s' must contain 6 sensors (Fx,Fy,Fz,Mx,My,Mz)" % name
            assert (np.atleast_1d((eval(str(nrs)))).shape[0] == 6), msg
        return Sensors(sensor_df)


class DTU_IEC64100_1_Ref_DLB(DLB):
    def __init__(self, iec_wt_class, Vr, D, z_hub, Vin=4, Vout=26, Vstep=2, T_turb=600, T_steady=100, years=20,
                 seed=None):
        """
        IEC 64100-1 3th edition (2005) with DTU interpretation described in
        Hansen, M. H., Thomsen, K., Natarajan, A., & Barlas, A. (2015). Design Load Basis for onshore turbines -
        Revision 00. DTU Wind Energy. DTU Wind Energy E No. 0074(EN)


        Parameters
        ----------
        iec_wt_class : str
            Wind turbine class "<wind class, 1-3><turbulence category a-c>"
        vr : int or float
            Rated wind speed [m/s]
        D : int or float
            Wind turbine rotor diameter [m]
        z_hub : int or float
            Hub height [m]
        Vin : int, optional
            Cut-in wind speed [m/s]. Default is 4 m/s
        Vout : int, optional
            Cut-out wind speed [m/s]. Default is 26 m/s
        Vstep : int, optional
            Wind speed step [m/s]
        T_turb : int, optional
            Simulation time [s] included in the analysis for DLCs with turbulent wind. Default is 600s
        T_steady : int, optional
            Simulation time [s] included in the analysis for DLCs with steady wind. Default is 100s
        years : int, optional
            Years of operation in fatigue load calculation. Default is 20 years
        Seed : int or None, optional
            If None, default, the seed numbers will be a continuous list, e.g. 6 seeds with s0=2300: 2300,2301,...2305
            If int, the Seed will be the input seed to a random generator that generates the seed number 0..9999
        """

        dlc_definitions = [
            dict(Name='DLC12', Description='Normal production', WSP='Vin:Vstep:Vout', Wdir='-10/0/10',
                 Turb='NTM', Seeds=6, Shear='NWP', Gust=None, Fault=None, Time=T_turb, Fatigue=True, Ultimate=True,
                 Name_dist="97.5", WSP_dist="Weibull", Wdir_dist="25/50/25"),
            dict(Name='DLC13', Description='Normal production with high turbulence', WSP='Vin:Vstep:Vout', Wdir='-10/0/10',
                 Turb='ETM', Seeds=6, Shear='NWP', Gust=None, Fault=None, Time=T_turb),
            dict(Name='DLC14', Description='Normal production with gust and direction change', WSP='Vr/Vr+2/Vr-2', Wdir=0,
                 Turb=None, Seeds=None, Shear='NWP', Gust='ECD', Fault=None, Time=T_steady),
            dict(Name='DLC15', Description='Normal production with extreme wind shear', WSP='Vin:Vstep:Vout', Wdir=0,
                 Turb=None, Seeds=None, Shear='EWS', Gust=None, Fault=None, Time=T_steady),
            dict(Name='DLC21', Description='Loss of electical network', WSP='Vin:Vstep:Vout', Wdir='-10/0/10',
                 Turb='NTM', Seeds=4, Shear='NWP', Gust=None, Fault='GridLoss10', Time=T_steady),
            dict(Name='DLC22y', Description='Abnormal yaw error', WSP='Vin:Vstep:Vout', Wdir='15:15:345',
                 Turb='NTM', Seeds=1, Shear='NWP', Gust=None, Fault=None, Time=T_turb),

            dict(Name='DLC24', Description='Power production with large yaw errors', WSP='Vin:Vstep:Vout', Wdir='-20/20',
                 Turb='NTM', Seeds=3, Shear='NWP', Gust=None, Fault=None, Time=T_turb, Fatigue=True, Ultimate=False,
                 Name_dist=50 / 24 / 365 * 100, WSP_dist="Weibull", Wdir_dist="50/50"),
            dict(Name='DLC31', Description='Start-up in normal wind profile', WSP='Vin/Vr/Vout', Wdir='0',
                 Turb=None, Seeds=None, Shear='NWP', Gust=None, Fault=None, Time=T_steady, Fatigue=True, Ultimate=False,
                 Name_dist="#1100", WSP_dist="#1000/#50/#50", Wdir_dist="0"),
            dict(Name='DLC41', Description='Shut-down in normal wind profile', WSP='Vin/Vr/Vout', Wdir='0',
                 Turb=None, Seeds=None, Shear='NWP', Gust=None, Fault=None, Time=T_steady, Fatigue=True, Ultimate=False,
                 Name_dist="#1100", WSP_dist="#1000/#50/#50", Wdir_dist="0"),
            dict(Name='DLC64', Description='Parked', WSP='4:2:0.7*Vref', Wdir='-8/8',
                 Turb='NTM', Seeds=6, Shear='NWP', Gust=None, Fault=None, Time=T_turb, Fatigue=True, Ultimate=False,
                 Name_dist="2.5", WSP_dist="Weibull", Wdir_dist="50/50"),
        ]

        variables = {'iec_wt_class': iec_wt_class, 'Vin': Vin,
                     'Vout': Vout, 'Vr': Vr, 'D': D, 'z_hub': z_hub, 'Vstep': Vstep,
                     'years': years}
        if seed:
            variables["seed"] = int(seed)
        else:
            variables["seed"] = seed
        DLB.__init__(self, dlc_definitions, variables)


def main():
    if __name__ == '__main__':

        dlb = DTU_IEC64100_1_Ref_DLB(iec_wt_class='1A',
                                     Vin=4, Vout=25, Vr=8,  # cut-in, cut_out and rated wind speed
                                     Vstep=4,
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
