import warnings
from wetb.gtsdf.unix_time import from_unix
from wetb.utils.postprocs import get_statistics
try:
    import h5py
except ImportError as e:
    raise ImportError("HDF5 library cannot be loaded. Windows XP is a known cause of this problem\n%s" % e)
import os
import numpy as np
import numpy.ma as ma
import xarray as xr
import glob
import tqdm
import inspect

block_name_fmt = "block%04d"


def load(filename, dtype=None):
    """Load a 'General Time Series Data Format'-hdf5 datafile

    Parameters
    ----------
    filename : str or h5py.File
        filename or open file object

    dtype: data type, optional
        type of returned data array, e.g. float16, float32 or float64.
        If None(default) the type of the returned data depends on the type of the file data

    Returns
    -------
    time : ndarray(dtype=float64), shape (no_observations,)
        time
    data : ndarray(dtype=dtype), shape (no_observations, no_attributes)
        data
    info : dict
        info containing:
            - type: "General Time Series Data Format"
            - name: name of dataset or filename if not present in file
            - no_attributes: Number of attributes
            - no_blocks: Number of datablocks
            - [description]: description of dataset or "" if not present in file
            - [attribute_names]: list of attribute names
            - [attribute_units]: list of attribute units
            - [attribute_descriptions]: list of attribute descriptions

    See Also
    --------
    gtsdf, save


    Examples
    --------
    >>> import gtsdf
    >>> data = np.arange(6).reshape(3,2)
    >>> gtsdf.save('test.hdf5', data)
    >>> time, data, info = gtsdf.load('test.hdf5')
    >>> print time
    [ 0.  1.  2.]
    >>> print data
    [[ 0.  1.]
     [ 2.  3.]
     [ 4.  5.]]
    >>> print info
    {'no_blocks': 1, 'type': 'General time series data format', 'name': 'test', 'no_attributes': 2, 'description': ''}
    >>> gtsdf.save('test.hdf5', data, name='MyDataset',
                                      description='MyDatasetDescription',
                                      attribute_names=['Att1', 'Att2'],
                                      attribute_units=['m', "m/s"],
                                      attribute_descriptions=['Att1Desc', 'Att2Desc'],
                                      time = np.array([0,1,4]),
                                      time_start = 10,
                                      time_step=2,
                                      dtype=np.float64)
    >>> time, data, info = gtsdf.load('test.hdf5')
    >>> print time
    [ 10.  12.  18.]
    >>> print data
    [[ 0.  1.]
     [ 2.  3.]
     [ 4.  5.]]
    >>> print info
    {'attribute_names': array(['Att1', 'Att2'], dtype='|S4'),
     'attribute_units': array(['m', 'm/s'], dtype='|S3'),
     'attribute_descriptions': array(['Att1Desc', 'Att2Desc'], dtype='|S8'),
     'name': 'MyDataset',
     'no_attributes': 2,
     'no_blocks': 1,
     'type': 'General time series data format',
     'description': 'MyDatasetDescription'}
    """
    f = _open_h5py_file(filename)
    try:
        info = _load_info(f)
        time, data = _load_timedata(f, dtype)
        return time, data, info
    finally:
        try:
            f.close()
        except BaseException:
            pass


def _open_h5py_file(filename):
    if isinstance(filename, h5py.File):
        f = filename
        filename = f.filename
    else:
        assert os.path.isfile(filename), "File, %s, does not exists" % filename
        f = h5py.File(filename, 'r')
    return f


def decode(v):
    if isinstance(v, bytes):
        return v.decode('latin1')
    elif hasattr(v, 'len'):
        return [decode(v_) for v_ in v]
    return v


def _load_info(f):

    info = {k: decode(v) for k, v in f.attrs.items()}
    check_type(f)
    if 'name' not in info:
        info['name'] = os.path.splitext(os.path.basename(f.filename))[0]
    if 'attribute_names' in f:
        info['attribute_names'] = [v.decode('latin1') for v in f['attribute_names']]
    if 'attribute_units' in f:
        info['attribute_units'] = [v.decode('latin1') for v in f['attribute_units']]
    if 'attribute_descriptions' in f:
        info['attribute_descriptions'] = [v.decode('latin1') for v in f['attribute_descriptions']]
    try:
        info['dtype'] = f[block_name_fmt % 0]['data'].dtype
    except BaseException:
        pass
    return info


def _load_timedata(f, dtype):
    no_blocks = f.attrs['no_blocks']
    if (block_name_fmt % 0) not in f:
        raise ValueError("HDF5 file must contain a group named '%s'" % (block_name_fmt % 0))
    block0 = f[block_name_fmt % 0]
    if 'data' not in block0:
        raise ValueError("group %s must contain a dataset called 'data'" % (block_name_fmt % 0))
    _, no_attributes = block0['data'].shape

    if dtype is None:
        file_dtype = f[block_name_fmt % 0]['data'].dtype
        if "float" in str(file_dtype):
            dtype = file_dtype
        elif file_dtype in [np.int8, np.uint8, np.int16, np.uint16]:
            dtype = np.float32
        else:
            dtype = np.float64
    time = []
    data = []
    for i in range(no_blocks):

        try:
            block = f[block_name_fmt % i]
        except Error:
            continue
        no_observations, no_attributes = block['data'].shape
        block_time = (block.get('time', np.arange(no_observations))[:]).astype(np.float64)
        if 'time_step' in block.attrs:
            block_time *= block.attrs['time_step']
        if 'time_start' in block.attrs:
            block_time += block.attrs['time_start']
        time.extend(block_time)

        block_data = block['data'][:].astype(dtype)
        if "int" in str(block['data'].dtype):
            block_data[block_data == np.iinfo(block['data'].dtype).max] = np.nan

        if 'gains' in block:
            block_data *= block['gains'][:]
        if 'offsets' in block:
            block_data += block['offsets'][:]
        data.append(block_data)

    if no_blocks > 0:
        data = np.vstack(data)
    return np.array(time).astype(np.float64), np.array(data).astype(dtype)


def save(filename, data, **kwargs):
    """Save a 'General Time Series Data Format'-hdf5 datafile

    Additional datablocks can be appended later using gtsdf.append_block

    Parameters
    ----------
    filename : str
    data : array_like, shape (no_observations, no_attributes)
    name : str, optional
        Name of dataset
    description : str, optional
        Description of dataset
    attribute_names : array_like, shape (no_attributes,), optional
        Names of attributes
    attribute_units : array_like, shape (no_attributes,), optinoal
        Units of attributes
    attribute_descriptions : array_like, shape(no_attributes,), optional
        Descriptions of attributes
    time : array_like, shape (no_observations, ), optional
        Time, default is [0..no_observations-1]
    time_start : int or float, optional
        Time offset (e.g. start time in seconds since 1/1/1970), default is 0, see notes
    time_step : int or float, optional
        Time scale factor (e.g. 1/sample frequency), default is 1, see notes
    dtype : data-type, optional
        Data type of saved data array, default uint16.\n
        Recommended choices:

        - uint16: Data is compressed into 2 byte integers using a gain and offset factor for each attribute
        - float64: Data is stored with high precision using 8 byte floats

    Notes
    -----
    Time can be specified by either

    - time (one value for each observation). Required inhomogeneous time distributions
    - time_start and/or time_step (one or two values), Recommended for homogeneous time distributions
    - time and time_start and/or time_step (one value for each observation + one or two values)

    When reading the file, the returned time-array is calculated as time * time_step + time_start

    See Also
    --------
    gtsdf, append_block, load


    Examples
    --------
    >>> import gtsdf
    >>> data = np.arange(12).reshape(6,2)
    >>> gtsdf.save('test.hdf5', data)
    >>> gtsdf.save('test.hdf5', data, name='MyDataset',
                                      description='MyDatasetDescription',
                                      attribute_names=['Att1', 'Att2'],
                                      attribute_units=['m', "m/s"],
                                      attribute_descriptions=['Att1Desc', 'Att2Desc'],
                                      time = np.array([0,1,2,6,7,8]),
                                      time_start = 10,
                                      time_step=2,
                                      dtype=np.float64)
    """
    if not filename.lower().endswith('.hdf5'):
        filename += ".hdf5"
    # exist_ok does not exist in Python27
    if not os.path.exists(os.path.dirname(os.path.abspath(filename))):
        os.makedirs(os.path.dirname(os.path.abspath(filename)))  # , exist_ok=True)
    _save_info(filename, data.shape, **kwargs)
    append_block(filename, data, **kwargs)


def _save_info(filename, data_shape, **kwargs):

    f = h5py.File(filename, "w")
    try:
        f.attrs["type"] = "General time series data format"
        no_observations, no_attributes = data_shape

        if 'name' in kwargs:
            f.attrs['name'] = kwargs['name']
        if 'description' in kwargs:
            f.attrs['description'] = kwargs['description']
        f.attrs['no_attributes'] = no_attributes
        if 'attribute_names' in kwargs:
            if no_attributes:
                assert len(kwargs['attribute_names']) == no_attributes, "len(attribute_names)=%d but data shape is %s" % (
                    len(kwargs['attribute_names']), data_shape)
            f.create_dataset("attribute_names", data=np.array([v.encode('utf-8') for v in kwargs['attribute_names']]))
        if 'attribute_units' in kwargs:
            if no_attributes:
                assert(len(kwargs['attribute_units']) == no_attributes)
            f.create_dataset("attribute_units", data=np.array([v.encode('utf-8') for v in kwargs['attribute_units']]))
        if 'attribute_descriptions' in kwargs:
            if no_attributes:
                assert(len(kwargs['attribute_descriptions']) == no_attributes)
            f.create_dataset("attribute_descriptions", data=np.array(
                [v.encode('utf-8') for v in kwargs['attribute_descriptions']]))
        f.attrs['no_blocks'] = 0
    except Exception:
        raise
    finally:
        f.close()


def append_block(filename, data, **kwargs):
    """Append a data block and corresponding time data to already existing file

    Parameters
    ----------
    filename : str
    data : array_like, shape (no_observations, no_attributes)
    time : array_like, shape (no_observations, ), optional
        Time, default is [0..no_observations-1]
    time_start : int or float, optional
        Time offset (e.g. start time in seconds since 1/1/1970), default is 0, see notes
    time_step : int or float, optional
        Time scale factor (e.g. 1/sample frequency), default is 1, see notes
    dtype : data-type, optional
        Data type of saved data array, default uint16.\n
        Recommended choices:

        - uint16: Data is compressed into 2 byte integers using a gain and offset factor for each attribute
        - float64: Data is stored with high precision using 8 byte floats

    Notes
    -----
    Time can be specified by either

    - time (one value for each observation). Required inhomogeneous time distributions
    - time_start and/or time_step (one or two values), Recommended for homogeneous time distributions
    - time and time_start and/or time_step (one value for each observation + one or two values)

    When reading the file, the returned time-array is calculated as time * time_step + time_start

    See Also
    --------
    gtsdf, save


    Examples
    --------
    >>> import gtsdf
    >>> data = np.arange(12).reshape(6,2)
    >>> gtsdf.save('test.hdf5', data)
    >>> gtsdf.append_block('test.hdf5', data+6)
    >>> time, data, info = gtsdf.load('test.hdf5')
    >>> print time
    [ 0.  1.  2.  3.  4.  5.]
    >>> print data
    [[  0.   1.]
     [  2.   3.]
     [  4.   5.]
     [  6.   7.]
     [  8.   9.]
     [ 10.  11.]]
    >>> print info
    {'no_blocks': 2, 'type': 'General time series data format', 'name': 'test', 'no_attributes': 2}
    """

    try:
        f = h5py.File(filename, "a")
        check_type(f)
        no_observations, no_attributes = data.shape
        assert(no_attributes == f.attrs['no_attributes'])
        blocknr = f.attrs['no_blocks']
        if blocknr == 0:
            dtype = kwargs.get('dtype', np.uint16)
        else:
            dtype = f[block_name_fmt % 0]['data'].dtype
            if dtype == np.uint16:
                if no_observations < 12:  # size with float32<1.2*size with uint16
                    dtype = np.float32

        block = f.create_group(block_name_fmt % blocknr)
        if 'time' in kwargs:
            assert(len(kwargs['time']) == no_observations)
            block.create_dataset('time', data=kwargs['time'])
        if 'time_step' in kwargs:
            time_step = kwargs['time_step']
            block.attrs['time_step'] = np.float64(time_step)
        if 'time_start' in kwargs:
            block.attrs['time_start'] = np.float64(kwargs['time_start'])

        pct_res = np.array([1])
        if "int" in str(dtype):
            if np.any(np.isinf(data)):
                f.close()
                raise ValueError(
                    "Int compression does not support 'inf'\nConsider removing outliers or use float datatype")
            nan = np.isnan(data)
            non_nan_data = ma.masked_array(data, nan)
            offsets = np.min(non_nan_data, 0)
            try:
                data = np.copy(data).astype(np.float64)
            except MemoryError:
                data = np.copy(data)
            data -= offsets
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ignore warning caused by abs(nan) and np.nanmax(nan)
                pct_res = (np.percentile(data[~np.isnan(data)], 75, 0) - np.percentile(data[~np.isnan(data)],
                                                                                       25, 0)) / np.nanmax(np.abs(data), 0)  # percent of resolution for middle half of data
            gains = np.max(non_nan_data - offsets, 0).astype(np.float64) / \
                (np.iinfo(dtype).max - 1)  # -1 to save value for NaN
            not0 = np.where(gains != 0)
            data[:, not0] /= gains[not0]

            data = data.astype(dtype)
            data[nan] = np.iinfo(dtype).max

            block.create_dataset('gains', data=gains)
            block.create_dataset('offsets', data=offsets)

        block.create_dataset("data", data=data.astype(dtype))
        f.attrs['no_blocks'] = blocknr + 1
        f.close()

        if "int" in str(dtype):
            int_res = (np.iinfo(dtype).max - np.iinfo(dtype).min)
            with np.errstate(invalid='ignore'):
                if min(pct_res[pct_res > 0]) * int_res < 256:
                    raise Warning("Less than 256 values are used to represent 50%% of the values in column(s): %s\nConsider removing outliers or use float datatype" % np.where(
                        pct_res[pct_res > 0] * int_res < 256)[0])

    except Exception:
        try:
            f.close()
        except BaseException:
            pass
        raise


def load_pandas(filename, dtype=None):
    import pandas as pd
    time, data, info = load(filename, dtype)
    df = pd.DataFrame()
    df["Time"] = time
    df["Date"] = [from_unix(t) for t in time]
    for n, d in zip(info['attribute_names'], data.T):
        df[n] = d
    return df


def check_type(f):
    if 'type' not in f.attrs or \
            (f.attrs['type'].lower() != "general time series data format" and f.attrs['type'].lower() != b"general time series data format"):
        raise ValueError("HDF5 file must contain a 'type'-attribute with the value 'General time series data format'")
    if 'no_blocks' not in f.attrs:
        raise ValueError("HDF5 file must contain an attribute named 'no_blocks'")


def remove_time_series(file):
    f = h5py.File(file, 'a')
    for group in f:
        if group.startswith('block'):
            del f[group]
    f.close()

    
def get_postproc(postproc_function, file, time_data_info=None, **kwargs):
    if time_data_info is None:
        time_data_info = load(file)
    time, data, info = time_data_info
    postproc_function_args = inspect.signature(postproc_function).parameters.keys()
    file_args = {}
    for item in ['file', 'time', 'data', 'info']:
        if item in postproc_function_args:
            file_args[item] = locals()[item]
    outputs = postproc_function(**file_args, **kwargs)
    return outputs


def write_postproc(file, postproc_outputs):
    if postproc_outputs is None:
        return
    if 'Postproc' not in file:
        file.create_group('Postproc')
    for group in postproc_outputs.keys():
        if group == 'data_array_info':
            continue
        if group in file['Postproc']:
            del file['Postproc'][group]
        file['Postproc'].create_group(group)
        for label, data in postproc_outputs[group].items():
            if label in file['Postproc'][group]:
                del file['Postproc'][group][label]        
            file['Postproc'][group].create_dataset(label, data=data)


def add_postproc(file, config={get_statistics: {'statistics': ['min', 'mean', 'max', 'std', 'eq3', 'eq4', 'eq6', 'eq8', 'eq10', 'eq12']}}):
    time_data_info = load(file)
    f = h5py.File(file, "a")
    data_array_info = {}
    for postproc, kwargs in config.items():
        postproc_outputs = get_postproc(postproc_function=postproc, file=file, time_data_info=time_data_info, **kwargs)
        write_postproc(file=f, postproc_outputs=postproc_outputs)
        if postproc_outputs is not None:
            if "data_array_info" in postproc_outputs.keys():
                data_array_info[list(postproc_outputs.keys())[0]] = postproc_outputs['data_array_info']
    f.close()
    return data_array_info
    

def load_postproc(filename, config={get_statistics: {'statistics': ['min', 'mean', 'max', 'std', 'eq3', 'eq4', 'eq6', 'eq8', 'eq10', 'eq12']}}, xarray=True, data_array_info=None):
    f = _open_h5py_file(filename)
    info = _load_info(f)
    if 'Postproc' not in f:
        print(f"Calculating postproc for '{filename}'")
        f.close()
        data_array_info = add_postproc(filename, config)
        return load_postproc(filename, config, xarray=xarray, data_array_info=data_array_info)
    if xarray:
        if data_array_info is None:
            f.close()
            data_array_info = add_postproc(filename, config)
            f = _open_h5py_file(filename)
        data_arrays = {}
        for group in f['Postproc']:
            if group in data_array_info.keys():
                if 'coords_function' in data_array_info[group].keys():
                    coords_function = data_array_info[group]['coords_function'](info, config)
                    if 'coords' in data_array_info[group].keys():
                        data_array_info[group]['coords'] = {**data_array_info[group]['coords'], **coords_function}
                    else:
                        data_array_info[group]['coords'] = coords_function
                    data_array_info[group].pop('coords_function')                     
                data_arrays[group] = xr.DataArray(data=f['Postproc'][group]['data'][:], **data_array_info[group])
            else:
                data_arrays[group] = xr.DataArray(data=f['Postproc'][group]['data'][:])
        f.close()
        return data_arrays
    else:
        dicts = {}
        for group in f['Postproc']:
            dicts[group] = {}
            for dataset in f['Postproc'][group]:
                dicts[group][dataset] = f['Postproc'][group][dataset][:]
        f.close()
        return dicts

 
def compress2postproc(filename, config={get_statistics: {'statistics': ['min', 'mean', 'max', 'std', 'eq3', 'eq4', 'eq6', 'eq8', 'eq10', 'eq12']}}):
    if remove_time_series not in config.keys():
        config[remove_time_series] = {}
    add_postproc(filename, config)


def collect_postproc(folder, root='.', filename='*.hdf5', recursive=True, config={get_statistics: {'statistics': ['min', 'mean', 'max', 'std', 'eq3', 'eq4', 'eq6', 'eq8', 'eq10', 'eq12']}}):
    if recursive:
        p = os.path.join(root, folder, '**', filename)
    else:
        p = os.path.join(root, folder, filename)
    fn_lst = glob.glob(p, recursive=recursive)
    if not fn_lst:
        raise Exception(f'No {filename} files found in {os.path.abspath(os.path.join(root, folder))}')
    postproc_output = [load_postproc(fn, config=config, xarray=True) for fn in tqdm.tqdm(sorted(fn_lst))]
    data_arrays = {}
    for postproc in postproc_output[0].keys():
        data = np.array([f[postproc] for f in postproc_output])
        dims = ('filename',) + postproc_output[0][postproc].dims
        coords = postproc_output[0][postproc].coords
        data_arrays[postproc] = xr.DataArray(data=data, dims=dims, coords=coords)
        data_arrays[postproc].coords['filename'] = [os.path.relpath(fn, root) for fn in fn_lst]
    return data_arrays

