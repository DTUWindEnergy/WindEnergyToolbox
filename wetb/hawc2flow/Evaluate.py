import pandas as pd
import numpy as np
import re, os
import click
from scipy import signal
from functools import wraps 
from wetb.hawc2.Hawc2io import ReadHawc2
from wetb.fatigue_tools.fatigue import eq_load

    
def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


class HAWC2DataFrame(pd.DataFrame):
    """
    A subclass of a pandas.DataFrame. The HAWC2DataFrame is able to link to a
    directory of HAWC2 result files. The HAWC2DataFrame contains an Accessor,
    named 'wetb', which contains functions to  populate the HAWC2DataFrame with
    post-processed statistics, and to aggregate the statistics.
    """

    _metadata = ['_fields', 'channels', '_directory', '_filenames']
    
    @property
    def _constructor(self):
        return HAWC2DataFrame
        
        
    def __init__(self, *args, dir=None, pattern=None, channels=None, **kwargs):
        
        if all(x is not None for x in [dir, channels]):            
            if pattern is None:
                res = self.link_directory(dir, channels)
            else:
                res = self.link_results(dir, pattern, channels)
            super(HAWC2DataFrame, self).__init__(res)

        else:
            super(HAWC2DataFrame, self).__init__(*args, **kwargs)


    
    def __finalize__(self, other, method=None, **kwargs):
        """propagate metadata from other to self """
        # merge operation: using metadata of the left object
        if method == 'merge':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.left, name, None))
        # concat operation: using metadata of the first object
        elif method == 'concat':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.objs[0], name, None))
        else:
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other, name, None))
        return self

    
    def link_results(self, directory, pattern_string, channels):
        self._directory = directory
        pattern, self._fields = self._compile_pattern(pattern_string)

        # get all filenames that fit the pattern
        self._filenames = [x[:-4] for x in os.listdir(directory) if x.endswith('.sel')]
        self._filenames = [x for x in self._filenames if pattern.match(x)]
        
        self.channels = channels
        # Extract input attributes and put in dataframe
        dat = []
        for fn in self._filenames:
            dat.append([float(x) if isFloat(x) else x for x in pattern.findall(fn)[0]])
        dat = pd.DataFrame(dat)
        
        # set column multi index
        if not dat.empty:
            column_tuples = list(zip(*[self._fields, ['']*len(self._fields)]))
            dat.columns = pd.MultiIndex.from_tuples(column_tuples, names=['channel', 'stat'])
        return dat
    
    
    def link_directory(self, directory, channels):
        '''
        Links all result files in a directory. Used if no pattern is given.
        '''
        self._directory = directory
        self.channels = channels

        self._fields = ['filename']
        # get all filenames of result files
        self._filenames = [x[:-4] for x in os.listdir(directory) if x.endswith('.sel')]
        dat = pd.DataFrame(self._filenames)

        # set column multi index
        if not dat.empty:
            column_tuples = list(zip(*[self._fields, ['']*len(self._fields)]))
            dat.columns = pd.MultiIndex.from_tuples(column_tuples, names=['channel', 'stat'])
        return dat
        
            
    @staticmethod
    def _compile_pattern(pattern_string):
        brackets = re.compile('{(.*?)}')
        fields = brackets.findall(pattern_string)
        for field in fields:
            pattern_string = pattern_string.replace('{'+ field +'}', '(.*)')
        return re.compile(pattern_string), fields   
        
                 
    def __call__(self, **kwargs):
        return self[self._mask( **kwargs)]


    def _mask(self, **kwargs):
        '''
        Returns a mask for refering to a dataframe, or self.Data, or self.Data_f, etc.
        example. dlc.mask(wsp=[12, 14], controller='noIPC')
        '''
        N = len(self)
        mask = [True] * N
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                mask_temp = [False] * N
                for v in value:
                    mask_temp = mask_temp | (self[key] == v)
                mask = mask & mask_temp
            else: #scalar, or single value
                mask = mask & (self[key] == value)
        return mask



@pd.api.extensions.register_dataframe_accessor("wetb")
class wetbAccessor(object):
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
    
        
    @staticmethod
    def _validate(obj):
        # verify that a HAWC2DataFrame is using the wetb accessor.
        if not isinstance(obj, HAWC2DataFrame):
            raise TypeError(f"wetb namespace is not accessible by type {type(obj)}. Expected: { type(HAWC2DataFrame())}")
            
            
    @classmethod    
    def populate_method(cls, method_name, label=None):
        label = label or method_name
        def decorator(func):
            @wraps(func) 
            def wrapper(self, channels, *args, **kwargs): 
                return self._add_stat(func, label, channels, *args, **kwargs)
            setattr(cls, method_name, wrapper)
            # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
            return func # returning func means func can still be used normally
        return decorator   
        
                 
    def fetch_result(self, idx, channels=None):
        '''
        Returns a time series dataframe of a single result file given an index number.
        '''
        fn = self._obj._filenames[idx]
        channels = channels or self.channels

        raw = ReadHawc2(os.path.join(self._obj._directory, fn)).ReadAll(ChVec=[i-1 for i in channels.values()])
        #raw = readHawc2Res(os.path.join(self._directory, fn), channels)
        return pd.DataFrame(raw, columns=channels.keys())
        # except:
        #     print(f'File {fn} could not be loaded.')
        
            
    def iter_sim(self, **kwargs):
        '''
        Iterates over simulation result files given the input filter defined in kwargs.
        '''
        sims_to_iterate = self._obj(**kwargs)
        for idx, row in sims_to_iterate.iterrows():
            raw = self.fetch_result(idx)
            yield row[self._obj._fields], raw

    
    def _add_stat(self, func, stat_name, channels=None, *args, **kwargs):
        '''
        Adds a column of statistics for the given channels using the given function.
        The function should take a pandas series (1d array) and return a float.
        '''
        values = []
        if channels is None:
            channels = self._obj.channels
        else:
            channels = {k:v for k,v in self._obj.channels.items() if k in channels}
        
        channel_string = ', '.join(channels)
        print(f'Calculating {stat_name} for {channel_string}...')
        
        with click.progressbar(self._obj.index) as bar:
            for idx in bar:
                raw = self.fetch_result(idx, channels)
                values.append(func(raw, *args, **kwargs))

        df = pd.DataFrame(values)
        # add multi index columns
        col_ch     = list(channels.keys())
        col_stat   = [stat_name]*len(col_ch)
        col_tuples = list(zip(*[col_ch, col_stat]))
        df.columns = pd.MultiIndex.from_tuples(col_tuples, names=['channel', 'stat'])

        return self._obj.join(df)


    def aggregate_rows(self, key):
        # used for taking the mean over all seeds
        print(f'Calculating mean over key={key}...')
        in_fields = [x for x in self._obj._fields if x != key]
        out_fields = [x for x in list(self._obj) if x[0] not in self._obj._fields]
        in_atts = self._obj[in_fields].drop_duplicates()
        new_dat = []
        with click.progressbar(in_atts.iterrows()) as bar:
            for _, x in bar:
                filt = {k[0]: v for k, v in dict(x).items()}
                new_dat.append(list(x) + list(self._obj(**filt)[out_fields].mean().values))
            
        # Make new HAWC2DataFrame and copy metadata (is there a better way?)
        new_df = HAWC2DataFrame(new_dat)
        for attr in self._obj._metadata:
            setattr(new_df, attr, getattr(self._obj, attr))
        new_df._fields = in_fields

        # add multi index columns
        col_ch     = in_fields + [x[0] for x in out_fields]
        col_stat   = ['']*len(in_fields) + [x[1] for x in out_fields]
        col_tuples = list(zip(*[col_ch, col_stat]))
        new_df.columns = pd.MultiIndex.from_tuples(col_tuples,
                    names=['channel', 'stat'])
        return new_df


    def aggregate_columns(self, source, dest, drop=False):
        # used for taking the mean over all blades.
        channel_string = ', '.join(source)
        print(f'Calculating mean over {channel_string}...')
        new_df = self._obj.copy()
        # get all unique stat indices
        col_stat = list(set(x[1] for x in list(self._obj)))
        for stat in col_stat:
            keys = [x for x in list(self._obj) if x[0] in source and x[1] == stat]
            if not keys:
                continue
            new_df[(dest, stat)] = self._obj[keys].mean(axis=1)
            
        if drop:
            # delete source columns
            new_df = new_df.drop(columns=source)
        return new_df

    
    @staticmethod
    def read_csv(filename):
        '''
        Reads a postproc csv file which was generated by the HAWC2Res class. Returns
        a DataFrame
        '''
        df           = HAWC2DataFrame(pd.read_csv(filename, header =[0, 1]))
        multicol     = list(df)
        multicol_new = []
        for col in multicol:
            if 'Unnamed' in col[1]:
                multicol_new.append((col[0], ''))
            else:
                multicol_new.append(col)
        df.columns = pd.MultiIndex.from_tuples(multicol_new)
        return df


@wetbAccessor.populate_method('DEL')
def DEL(x, m=4):
    '''
    Calculates the damage equivalent load of a set of time series loads.
    args:
        x (pd.DataFrame): load time series, where each column is a different load channel
        m (float): wohler constant (default: 4)
    returns:
        DEL (list of floats): Damage equivalent load for each load channel.
    '''
    DEL = []
    for k in list(x):
        DEL.append(eq_load(x[k].values, m=m)[0][0])
    return DEL
    
    
@wetbAccessor.populate_method('mean', label='Mean')
def _mean(x):
    '''
    Calculates the mean of a set of time series.
    args:
        x (pd.DataFrame): time series, where each column contains a different time series.
    returns:
        (list of floats): Mean of each time series.
    '''
    return x.mean().values
    
    
@wetbAccessor.populate_method('var', label='Var')
def _var(x):
    '''
    Calculates the variance of a set of time series.
    args:
        x (pd.DataFrame): time series, where each column contains a different time series.
    returns:
        (list of floats): variance of each time series.
    '''
    return x.var().values


@wetbAccessor.populate_method('std', label='Std')
def _std(x):
    '''
    Calculates the standard deviation of a set of time series.
    args:
        x (pd.DataFrame): time series, where each column contains a different time series.
    returns:
        (list of floats): standard deviation of each time series.
    '''
    return x.std().values


@wetbAccessor.populate_method('final')
def _final(x):
    '''
    Returns the final value of a set of time series.
    args:
        x (pd.DataFrame): time series, where each column contains a different time series.
    returns:
        (list of floats): final value of each time series.
    '''
    return x.iloc[-1].values


@wetbAccessor.populate_method('PSD')
def _psd(x, fs=100):
    '''
    Calculates power spectral density (PSD) of a set of time series.
    args:
        x (pd.DataFrame): time series, where each column contains a different time series.
        fs (float): sampling frequency in Hz (default: 100Hz).
    returns:
        (list of arrays): PSD of each time series.
    '''
    PSD = []
    for k in list(x):
        f, Py = signal.welch(x[k].values, fs=fs, nperseg=1024*8)
        PSD.append(Py)
    return PSD
            
