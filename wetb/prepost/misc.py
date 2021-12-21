# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:09:04 2012

Library for general stuff

@author: dave
"""

import os
import io
import unittest
import pickle
import re

import numpy as np
import scipy as sp
from scipy import optimize as opt
from scipy import stats
from scipy.interpolate import griddata as interp
try:
    from matplotlib import pyplot as plt
except:
    pass
import pandas as pd


class Logger(object):
    """The Logger class can be used to redirect standard output to a log file.
    Usage: Create a Logger object and redirect standard output to the Logger
    object.  For example:
    output = Logger(file_handle, True)
    import sys
    sys.stdout = output
    """

    def __init__(self, logFile, echo):
        """Arguments:
        logFile     a file object that is available for writing
        echo        Boolean.  If True, output is sent to standard output in
                    addition to the log file.
        """
        import sys
        self.out = sys.stdout
        self.logFile = logFile
        self.echo = echo

    def write(self, s):
        """Required method that replaces stdout. You don't have to call this
        directly--all print statements will be redirected here."""
        self.logFile.write(s)
        if self.echo:
            self.out.write(s)
        self.logFile.flush()


def path_split_dirs(path):
    """
    Return a list with dirnames. Ignore any leading "./"
    """
    dirs = path.split(os.path.sep)
    if dirs[0] == '.':
        dirs.pop(0)
    return dirs


def path_sanitize(path, allowdd=False, allowabs=False):
    """Raises a ValueError if not considered safe. A leading ../ is allowed
    when allowdd=True.
    """

    if not isinstance(path, str):
        raise ValueError('path_sanitize requires a string as input.')

    keepcharacters = set(['/', '.', '_', '-'])

    # in special cases we allow one leading ../
    if allowdd and path[:3] == '../':
        path = path[3:]

    # no absolute paths if not allowabs
    if path[0] == '/' and not allowabs:
        raise ValueError('Absolute paths not allowed: "%s"' % path)

    # no empty paths either
    if path == '':
        raise ValueError('Empty paths are not allowed')

    # only alphanummerical characters (includes unicode)
    if not all([c.isalnum() or c in keepcharacters for c in path]):
        raise ValueError('Invalid or unsafe path: "%s"' % path)

    # additional checks on sub-directories
    items = path.split('/')
    # leading/trailing '/' leads to first/last element being emtpy
    if path[0] == "/":
        items = items[1:]
    if path[-1] == '/':
        items = items[:-1]

    no_lt = set(['.', '-'])  # characters not allowed leading/trailing a sub-dir
    for item in items:
        # require a sub-dir to be at least 2 characters or more
        if len(item) < 2:
            msg = 'Directories and filenames need to be longer than 1 character.'
            raise ValueError('Invalid or unsafe path: "%s". %s' % (path, msg))
        if item[0] in no_lt or item[-1] in no_lt:
            msg = 'No leading/trailing . or - allowed.'
            raise ValueError('Invalid or unsafe path: "%s". %s' % (path, msg))


def sanitize_wine_prefix(wine_prefix):
    """special case to sanitize
    """

    # sanitize wine_prefix
    # for backwards compatibility
    if wine_prefix[:2] == '~/':
        wine_prefix = wine_prefix[2:]
    # first character can also be a dot
    if not wine_prefix[0].isalnum():
        if wine_prefix[0]=='.':
            pass
        else:
            raise ValueError('Invalid wine_prefix value: %s' % wine_prefix)
    if not wine_prefix[1:].isalnum():
        raise ValueError('Invalid wine_prefix value: %s' % wine_prefix)
    return os.path.join('$HOME', wine_prefix).replace("\\", "/")


def unique(s):
    """
    SOURCE: http://code.activestate.com/recipes/52560/
    AUTHOR: Tim Peters

    Return a list of the elements in s, but without duplicates.

    For example, unique([1,2,3,1,2,3]) is some permutation of [1,2,3],
    unique("abcabc") some permutation of ["a", "b", "c"], and
    unique(([1, 2], [2, 3], [1, 2])) some permutation of
    [[2, 3], [1, 2]].

    For best speed, all sequence elements should be hashable.  Then
    unique() will usually work in linear time.

    If not possible, the sequence elements should enjoy a total
    ordering, and if list(s).sort() doesn't raise TypeError it's
    assumed that they do enjoy a total ordering.  Then unique() will
    usually work in O(N*log2(N)) time.

    If that's not possible either, the sequence elements must support
    equality-testing.  Then unique() will usually work in quadratic
    time.
    """

    n = len(s)
    if n == 0:
        return []

    # Try using a dict first, as that's the fastest and will usually
    # work.  If it doesn't work, it will usually fail quickly, so it
    # usually doesn't cost much to *try* it.  It requires that all the
    # sequence elements be hashable, and support equality comparison.
    u = {}
    try:
        for x in s:
            u[x] = 1
    except TypeError:
        del u  # move on to the next method
    else:
        return list(u.keys())

    # We can't hash all the elements.  Second fastest is to sort,
    # which brings the equal elements together; then duplicates are
    # easy to weed out in a single pass.
    # NOTE:  Python's list.sort() was designed to be efficient in the
    # presence of many duplicate elements.  This isn't true of all
    # sort functions in all languages or libraries, so this approach
    # is more effective in Python than it may be elsewhere.
    try:
        t = list(s)
        t.sort()
    except TypeError:
        del t  # move on to the next method
    else:
        assert n > 0
        last = t[0]
        lasti = i = 1
        while i < n:
            if t[i] != last:
                t[lasti] = last = t[i]
                lasti += 1
            i += 1
        return t[:lasti]

    # Brute force is all that's left.
    u = []
    for x in s:
        if x not in u:
            u.append(x)
    return u


def CoeffDeter(obs, model):
    """
    Coefficient of determination
    ============================

    https://en.wikipedia.org/wiki/Coefficient_of_determination

    Parameters
    ----------

    obs : ndarray(n) or list
        The observed dataset

    model : ndarray(n), list or scalar
        The fitted dataset

    Returns
    -------

    R2 : float
        The coefficient of determination, varies between 1 for a perfect fit,
        and 0 for the worst possible fit ever

    """

    if type(obs).__name__ == 'list':
        obs = np.array(obs)

    SS_tot = np.sum(np.power( (obs - obs.mean()), 2 ))
    SS_err = np.sum(np.power( (obs - model), 2 ))
    R2 = 1 - (SS_err/SS_tot)

    return R2


def calc_sample_rate(time, rel_error=1e-4):
    """
    the sample rate should be constant throughout the measurement serie
    define the maximum allowable relative error on the local sample rate

    rel_error = 1e-4 # 0.0001 = 0.01%
    """
    deltas = np.diff(time)
    # the sample rate should be constant throughout the measurement serie
    # define the maximum allowable relative error on the local sample rate
    if not (deltas.max() - deltas.min())/deltas.max() <  rel_error:
        print('Sample rate not constant, max, min values:', end='')
        print('%1.6f, %1.6f' % (1/deltas.max(), 1/deltas.min()))
#        raise AssertionError
    return 1/deltas.mean()


def findIntersection(fun1, fun2, x0):
    """
    Find Intersection points of two functions
    =========================================

    Find the intersection between two random callable functions.
    The other alternative is that they are not callable, but are just numpy
    arrays describing the functions.

    Parameters
    ----------

    fun1 : calable
        Function 1, should return a scalar and have one argument

    fun2 : calable
        Function 2, should return a scalar and have one argument

    x0 : float
        Initial guess for sp.optimize.fsolve

    Returns
    -------



    """
    return sp.optimize.fsolve(lambda x : fun1(x) - fun2(x), x0)


# TODO: replace this with some of the pyrain functions
def find0(array, xi=0, yi=1, verbose=False, zerovalue=0.0):
    """
    Find single zero crossing
    =========================

    Find the point where a x-y dataset crosses zero. This method can only
    handle one zero crossing point.

    Parameters
    ----------
    array : ndarray
        should be 2D, with a least 2 columns and 2 rows

    xi : int, default=0
        index of the x values on array[:,xi]

    yi : int, default=1
        index of the y values on array[:,yi]

    zerovalue : float, default=0
        Set tot non zero to find the corresponding crossing.

    verbose : boolean, default=False
        if True intermediate results are printed. Usefull for debugging

    Returns
    -------
    y0 : float
        if no x0=0 exists, the result will be an interpolation between
        the two points around 0.

    y0i : int
        index leading to y0 in the input array. In case y0 was the
        result of an interpolation, the result is the one closest to x0=0

    """

    # Determine the two points where aoa=0 lies in between
    # take all the negative values, the maximum is the one closest to 0
    try:
        neg0i = np.abs(array[array[:,xi].__le__(zerovalue),xi]).argmax()
    # This method will fail if there is no zero crossing (not enough data)
    # in other words: does the given data range span from negative, to zero to
    # positive?
    except ValueError:
        print('Given data range does not include zero crossing.')
        return 0,0

    # find the points closest to zero, sort on absolute values
    isort = np.argsort(np.abs(array[:,xi]-zerovalue))
    if verbose:
        print(array[isort,:])
    # find the points closest to zero on both ends of the axis
    neg0i = isort[0]
    sign = int(np.sign(array[neg0i,xi]))
    # only search for ten points
    for i in range(1,20):
        # first time we switch sign, we have it
        if int(np.sign(array[isort[i],xi])) is not sign:
            pos0i = isort[i]
            break

    try:
        pos0i
    except NameError:
        print('Given data range does not include zero crossing.')
        return 0,0

    # find the value closest to zero on the positive side
#    pos0i = neg0i +1

    if verbose:
        print('0_negi, 0_posi', neg0i, pos0i)
        print('x[neg0i], x[pos0i]', array[neg0i,xi], array[pos0i,xi])

    # check if x=0 is an actual point of the series
    if np.allclose(array[neg0i,xi], 0):
        y0 = array[neg0i,yi]
        if verbose:
            prec = ' 01.08f'
            print('y0:', format(y0, prec))
            print('x0:', format(array[neg0i,xi], prec))
    # check if x=0 is an actual point of the series
    elif np.allclose(array[pos0i,xi], 0):
        y0 = array[pos0i,yi]
        if verbose:
            prec = ' 01.08f'
            print('y0:', format(y0, prec))
            print('x0:', format(array[pos0i,xi], prec))
    # if not very close to zero, interpollate to find the zero point
    else:
        y1 = array[neg0i,yi]
        y2 = array[pos0i,yi]
        x1 = array[neg0i,xi]
        x2 = array[pos0i,xi]
        y0 = (-x1*(y2-y1)/(x2-x1)) + y1

        if verbose:
            prec = ' 01.08f'
            print('y0:', format(y0, prec))
            print('y1, y2', format(y1, prec), format(y2, prec))
            print('x1, x2', format(x1, prec), format(x2, prec))

    # return the index closest to the value of AoA zero
    if abs(array[neg0i,0]) > abs(array[pos0i,0]):
        y0i = pos0i
    else:
        y0i = neg0i

    return y0, y0i


def remove_items(list, value):
    """Remove items from list
    The given list wil be returned withouth the items equal to value.
    Empty ('') is allowed. So this is een extension on list.remove()
    """
    # remove list entries who are equal to value
    ind_del = []
    for i in range(len(list)):
        if list[i] == value:
            # add item at the beginning of the list
            ind_del.insert(0, i)

    # remove only when there is something to remove
    if len(ind_del) > 0:
        for k in ind_del:
            del list[k]

    return list


class DictDB(object):
    """
    A dictionary based database class
    =================================

    Each tag corresponds to a row and each value holds another tag holding
    the tables values, or for the current row the column values.

    Each tag should hold a dictionary for which the subtags are the same for
    each row entry. Otherwise you have columns appearing and dissapearing.
    That is not how a database is expected to behave.
    """

    def __init__(self, dict_db):
        """
        """
        # TODO: data checks to see if the dict can qualify as a database
        # in this context

        self.dict_db = dict_db

    def search(self, dict_search):
        """
        Search a dictionary based database
        ==================================

        Searching on based keys having a certain value.

        Parameters
        ----------

        search_dict : dictionary
            Keys are the column names. If the values match the ones in the
            database, the respective row gets selected. Each tag is hence
            a unique row identifier. In case the value is a list (or it will
            be faster if it is a set), all the list entries are considered as
            a go.
        """
        self.dict_sel = dict()

        # browse through all the rows
        for row in self.dict_db:
            # and for each search value, check if the row holds the requested
            # column value
            init = True
            alltrue = True
            for col_search, val_search in list(dict_search.items()):
                # for backwards compatibility, convert val_search to list
                if not type(val_search).__name__ in ['set', 'list']:
                    # conversion to set is more costly than what you gain
                    # by target in set([]) compared to target in []
                    # conclusion: keep it as a list
                    val_search = [val_search]

                # all items should be true
                # if the key doesn't exists, it is not to be considered
                try:
                    if self.dict_db[row][col_search] in val_search:
                        if init or alltrue:
                            alltrue = True
                    else:
                        alltrue = False
                except KeyError:
                    alltrue = False
                init = False
            # all search criteria match, save the row
            if alltrue:
                self.dict_sel[row] = self.dict_db[row]

    # TODO: merge with search into a more general search/select method?
    # shouldn't I be moving to a proper database with queries?
    def search_key(self, dict_search):
        """
        Search for a string in dictionary keys
        ======================================

        Searching based on the key of the dictionaries, not the values

        Parameters
        ----------

        searchdict : dict
            As key the search string, as value the operator: True for inclusive
            and False for exclusive. Operator is AND.

        """

        self.dict_sel = dict()

        # browse through all the rows
        for row in self.dict_db:
            # and see for each row if its name contains the search strings
            init = True
            alltrue = True
            for col_search, inc_exc in dict_search.items():
                # is it inclusive the search string or exclusive?
                if (row.find(col_search) > -1) == inc_exc:
                    if init:
                        alltrue = True
                else:
                    alltrue = False
                    break
                init = False
            # all search criteria matched, save the row
            if alltrue:
                self.dict_sel[row] = self.dict_db[row]


class DictDiff(object):
    """
    Calculate the difference between two dictionaries as:
    (1) items added
    (2) items removed
    (3) keys same in both but changed values
    (4) keys same in both and unchanged values

    Source
    ------

    Basic idea of the magic is based on following stackoverflow question
    http://stackoverflow.com/questions/1165352/
    fast-comparison-between-two-python-dictionary
    """
    def __init__(self, current_dict, past_dict):
        self.current_d = current_dict
        self.past_d    = past_dict
        self.set_current  = set(current_dict.keys())
        self.set_past     = set(past_dict.keys())
        self.intersect    = self.set_current.intersection(self.set_past)
    def added(self):
        return self.set_current - self.intersect
    def removed(self):
        return self.set_past - self.intersect
    def changed(self):
        #set(o for o in self.intersect if self.past_d[o] != self.current_d[o])
        # which is the  similar (exept for the extension) as below
        olist = []
        for o in self.intersect:
            # if we have a numpy array
            if type(self.past_d[o]).__name__ == 'ndarray':
                if not np.allclose(self.past_d[o], self.current_d[o]):
                    olist.append(o)
            elif self.past_d[o] != self.current_d[o]:
                olist.append(o)
        return set(olist)

    def unchanged(self):
        t=set(o for o in self.intersect if self.past_d[o] == self.current_d[o])
        return t


def fit_exp(time, data, checkplot=True, method='linear', func=None, C0=0.0):
    """
    Note that all values in data have to be possitive for this method to work!
    """

    def fit_exp_linear(t, y, C=0):
        y = y - C
        y = np.log(y)
        K, A_log = np.polyfit(t, y, 1)
        A = np.exp(A_log)
        return A, K

    def fit_exp_nonlinear(t, y):
        # The model function, f(x, ...). It must take the independent variable
        # as the first argument and the parameters to fit as separate remaining
        # arguments.
        opt_parms, parm_cov = sp.optimize.curve_fit(model_func,t,y)
        A, K, C = opt_parms
        return A, K, C

    def model_func(t, A, K, C):
        return A * np.exp(K * t) + C

    # Linear fit
    if method == 'linear':
#        if data.min() < 0.0:
#            msg = 'Linear exponential fitting only works for positive values'
#            raise ValueError, msg
        A, K = fit_exp_linear(time, data, C=C0)
        fit = model_func(time, A, K, C0)
        C = C0

    # Non-linear Fit
    elif method == 'nonlinear':
        A, K, C = fit_exp_nonlinear(time, data)
        fit = model_func(time, A, K, C)

    if checkplot:
        plt.figure()
        plt.plot(time, data, 'ro', label='data')
        plt.plot(time, fit, 'b', label=method)
        plt.legend(bbox_to_anchor=(0.9, 1.1), ncol=2)
        plt.grid()

    return fit, A, K, C


def curve_fit_exp(time, data, checkplot=True, weights=None):
    """
    This code is based on a StackOverflow question/answer:
    http://stackoverflow.com/questions/3938042/
    fitting-exponential-decay-with-no-initial-guessing

    A*e**(K*t) + C
    """

    def fit_exp_linear(t, y, C=0):
        y = y - C
        y = np.log(y)
        K, A_log = np.polyfit(t, y, 1)
        A = np.exp(A_log)
        return A, K

    def fit_exp_nonlinear(t, y):
        # The model function, f(x, ...). It must take the independent variable
        # as the first argument and the parameters to fit as separate remaining
        # arguments.
        opt_parms, parm_cov = sp.optimize.curve_fit(model_func,t,y)
        A, K, C = opt_parms
        return A, K, C

    def model_func(t, A, K, C):
        return A * np.exp(K * t) + C

    C0 = 0

    ## Actual parameters
    #A0, K0, C0 = 2.5, -4.0, 0.0
    ## Generate some data based on these
    #tmin, tmax = 0, 0.5
    #num = 20
    #t = np.linspace(tmin, tmax, num)
    #y = model_func(t, A0, K0, C0)
    ## Add noise
    #noisy_y = y + 0.5 * (np.random.random(num) - 0.5)

    # Linear fit
    A_lin, K_lin = fit_exp_linear(time, data, C=C0)
    fit_lin = model_func(time, A_lin, K_lin, C0)

    # Non-linear Fit
    A_nonlin, K_nonlin, C = fit_exp_nonlinear(time, data)
    fit_nonlin = model_func(time, A_nonlin, K_nonlin, C)

    # and plot
    if checkplot:
        plt.figure()
        plt.plot(time, data, 'ro', label='data')
        plt.plot(time, fit_lin, 'b', label='linear')
        plt.plot(time[::-1], fit_nonlin, 'g', label='nonlinear')
        plt.legend(bbox_to_anchor=(0.9, 1.0), ncol=3)
        plt.grid()

    return


def readlines_try_encodings(fname):
    """Read text file in binary and try to encode with a few common encodings.
    Return the first succesful attempt.

    Parameters
    ----------
    fname : str
        Path to file to be read.

    Returns
    -------
    lines : list of str
        Returns same as open(fname).readlines().
    """

    # some potentially common encoding formats
    encodings = ('utf-8', 'cp1252', 'iso-8859-1', 'latin-1', 'cp1250',
                 'cp1251', 'cp1253', 'iso-8859-2', 'iso-8859-7', 'macgreek')

    # open the file and read entire object in as bytes, we do this to avoid
    # re-opening and reading the file for each encoding we try
    with open(fname, 'rb') as fid:
        fbytes = fid.read()

    for enc in encodings:
        # convert to a file object
        fid = io.BytesIO(fbytes)
        try:
            # we want to mimic the standard readlines behaviour
            text_wrapper = io.TextIOWrapper(fid, encoding=enc)
            lines = text_wrapper.readlines()
        except UnicodeDecodeError:
            print('encoding with', enc, 'does not work for:', fname)
            print('unicode error with %s , trying different encoding' % enc)
            pass
        else:
            # print('opening the file with encoding:  %s ' % enc)
            break
    return lines


def read_excel_files(proot, fext='xlsx', pignore=None, sheet=0,
                     pinclude=None, silent=False):
    """
    Read recursively all MS Excel files with extension "fext". Only the
    default name for the first sheet (Sheet1) of the Excel file is considered.

    Parameters
    ----------

    proot : string
        Path that will be recursively explored for the presence of files
        that have file extension "fext"

    fext : string, default='xlsx'
        File extension of the Excel files that should be loaded. Other valid
        extensions are csv, xls, and xlsm.

    pignore : string, default=None
        Specify which string can not occur in the full path of the DLC target.

    pinclude : string, default=None
        Specify which string has to occur in the full path of the DLC target.

    sheet : string or int, default=0
        Name or index of the Excel sheet to be considered. By default, the
        first sheet (index=0) is taken. Ignored when fext is csv.

    Returns
    -------

    df_list : dictionary
        A dictionary with the Excel file name (excluding 'fext') as key, and
        the corresponding pandas DataFrame as value.

    """

    df_list = {}
    # find all dlc defintions in the subfolders
    for root, dirs, files in os.walk(proot):
        for file_name in files:
            if not file_name.split('.')[-1] == fext:
                continue
            f_target = os.path.join(root, file_name)
            # if it does not contain pinclude, ignore the dlc
            if pinclude is not None and f_target.find(pinclude) < 0:
                continue
            # if it does contain pignore, ingore the dlc
            if pignore is not None and f_target.find(pignore) > -1:
                continue
            if not silent:
                print(f_target, end='')
            try:
                if fext == 'csv':
                    df = pd.read_csv(f_target)
                else:
                    df = pd.read_excel(f_target, sheet_name=sheet)
                df_list[f_target.replace('.'+fext, '')] = df
                if not silent:
                    print(': sucesfully included %i case(s)' % len(df))
            except:
                if not silent:
                    print('     XXXXX ERROR COULD NOT READ')

    return df_list


def check_df_dict(df_dict):
    """
    Verify if the dictionary that needs to be transferred to a Pandas DataFrame
    makes sense

    Returns
    -------

    collens : dict
        Dictionary with df_dict keys as keys, len(df_dict[key]) as column.
        In other words: the length of each column (=rows) of the soon to be df.
    """
    collens = {}
    for col, values in df_dict.items():
        print('%6i : %s' % (len(values), col))
        collens[col] = len(values)
    return collens


def find_tags(fname):
    """
    Find all unqiue tags in a text file.
    """

    with open(fname, 'r') as f:
        lines = f.readlines()

    # regex for finding all tags in a line
    regex = re.compile('(\\[.*?\\])')
    tags_in_master = {}

    for i, line in enumerate(lines):
        # are there any tags on this line? Ignore comment AND label section
        tags = regex.findall(line.split(';')[0].split('#')[0])
        for tag in tags:
            try:
                tags_in_master[tag].append(i)
            except KeyError:
                tags_in_master[tag] = [i]

    return tags_in_master


def read_mathematica_3darray(fname, shape=None, data=None, dtype=None):
    """
    I am not sure with which Mathematica command you generate this data,
    but this is the format in which I got it.

    Parameters
    ----------

    fname : str

    shape : tuple, default=None
        Tuple with 3 elements, defining the ndarray elements for each of the
        axes. Only used when data is set to None.

    dtype : dtype, default=None
        Is used to set the data dtype when data=None.

    data : ndarray, default=None
        When None, the data array is created according to shape and dtype.

    Returns
    -------

    data : ndarray

    """

    if data is None:
        data = np.ndarray(shape, dtype=dtype)
    else:
        dtype = data.dtype
    with open(fname, 'r') as f:
        for i, line in enumerate(f.readlines()):
            els = line.split('}","{')
            for j, row in enumerate(els):
                row_ = row.replace('{', '').replace('}', '').replace('"', '')
                data[i,j,:] = np.genfromtxt(row_.split(', '), dtype=dtype)

    return data


def CDF(series, sort=True):
    """
    Cumulative distribution function
    ================================

    Cumulative distribution function of the form:

    .. math::
        CDF(i) = \\frac{i-0.3}{N - 0.9}

    where
        i : the index of the sorted item in the series
        N : total number of elements in the serie
    Series will be sorted first.

    Parameters
    ----------
    series : ndarra(N)

    sort  : bool, default=True
        to sort or not to sort

    Returns
    -------
    cdf : ndarray (N,2)
        Array with the sorted input series on the first column
        and the cumulative distribution function on the second.

    """

    N = len(series)
    # column array
    i_range = np.arange(N)
    # convert to row array
    x, i_range = np.meshgrid([1], i_range)
    # to sort or not to sort the input series
    if sort:
        series.sort(axis=0)
    # convert to row array. Do after sort, otherwise again 1D column array
    x, series = np.meshgrid([1], series)
    # cdf array
    cdf = sp.zeros((N,2))
    # calculate the actual cdf values
    cdf[:,1] = (i_range[:,0]-0.3)/(float(N)-0.9)
    # make sure it is sorted from small to large
    if abs(series[0,0]) > abs(series[series.shape[0]-1,0]) and series[0,0] < 0:
        # save in new variable, otherwise things go wrong!!
        # if we do series[:,0] = series[::-1,0], we get somekind of mirrord
        # array
        series2 = series[::-1,0]
    # x-channel should be on zero for plotting
    cdf[:,0] = series2[:]

    return cdf


def rebin(hist, bins, nrbins):
    """
    Assume within a bin, the values are equally distributed. Only works for
    equally spaced bins.
    """

    binrange = float(bins[-1] - bins[0])
    width = np.diff(bins).mean()
    width_ = binrange / float(nrbins)
    hist_ = sp.zeros((nrbins))
    bins_ = np.linspace(bins[0], bins[-1], num=nrbins+1)

    if width_ < width:
        raise ValueError('you can only rebin to larger bins')

    if not len(hist)+1 == len(bins):
        raise ValueError('bins should contain the bin edges')

    window, j = width, 0
#    print('width:', width)
#    print('j=0')
    for i, k in enumerate(hist):
        if window < width_:
            hist_[j] += hist[i]#*width
#            print('window=%1.04f' % window, end=' ')
#            print('(%02i):%1.04f' % (i, hist[i]))
            window += width
            if i+1 == len(hist):
                print()
        else:
            w_right = (window - width_) / width
            w_left = (width - (window - width_)) / width
            hist_[j] += hist[i]*w_left
#            print('window=%1.04f' % window, end=' ')
#            print('(%02i):%1.04f*(%1.02f)' % (i, hist[i], w_left), end=' ')
#            print('T: %1.04f' % hist_[j])
            if j+1 >= nrbins:
                hist_[j] += hist[i]*w_right
                print('')
                return hist_, bins_
            j += 1
#            print('j=%i' % j)
#            print('window=%1.04f' % window, end=' ')
            hist_[j] += hist[i]*w_right
            window = w_right*width + width
#            print('(%02i):%1.04f*(%1.02f)' % (i, hist[i], w_right))
#    print('')
    return hist_, bins_


def histfit(hist, bin_edges, xnew):
    """
    This should be similar to the Matlab function histfit:
    http://se.mathworks.com/help/stats/histfit.html

    Based on:
    http://nbviewer.ipython.org/url/xweb.geos.ed.ac.uk/~jsteven5/blog/
    fitting_distributions_from_percentiles.ipynb

    Calculate the CDF of given PDF, and fit a lognorm distribution onto the
    CDF. This obviously only works if your PDF is lognorm.

    Parameters
    ----------

    hist : ndarray(n)

    bin_edges : ndarray(n+1)

    xnew : ndarray(k)

    Returns
    -------

    shape_out

    scale_out

    pdf_fit : ndarray(k)

    """

    # Take the upper edges of the bins. I tried to use the center of the bin
    # and the left bin edges, but it works best with the right edges
    # It only works ok with x data is positive, force only positive x-data
    x_hist = (bin_edges - bin_edges[0])[1:]
    y_hist = hist.cumsum()/hist.cumsum().max()  # Normalise the cumulative sum

    # FIT THE DISTRIBUTION
    (shape_out, scale_out), pcov = opt.curve_fit(
                lambda xdata, shape, scale: stats.lognorm.cdf(xdata, shape,
                loc=0, scale=scale), x_hist, y_hist)

    pdf_fit =  stats.lognorm.pdf(xnew, shape_out, loc=0, scale=scale_out)
    # normalize
    width = np.diff(x_hist).mean()
    pdf_fit = pdf_fit / (pdf_fit * width).sum()
    return shape_out, scale_out, pdf_fit


def histfit_arbritrary(edges, pdf, edges_new, resolution=100):
    """Re-bin based on the CDF of a PDF. Assume normal distribution within
    a bin to transform the CDF to higher resolution.

    Parameters
    ----------

    edges : ndarray(n+1)
        edges of the bins, inlcuding most left and right edges.

    pdf : ndarray(n)
        probability of the bins

    edges_new : ndarray(m+1)
        edges of the new bins

    resolution : int
        resolution of the intermediate CDF used for re-fitting.


    Returns
    -------

    centers_new : ndarray(m)

    widths_new : ndarray(m)

    pdf_new : ndarray(m)

    """

    x_hd = np.ndarray((0,))
    cdf_hd = np.ndarray((0,))
    binw = np.ndarray((0,))

    for i in range(len(pdf)):
        # HD grid for x
        x_inc = np.linspace(edges[i], edges[i+1], num=resolution)

        # FIXME: let the distribution in a bin be a user configurable input
        # define a distribution within the bin: norm
        shape = 2.5
        scale = shape*2/10
        x_inc = np.linspace(0, scale*10, num=resolution)
        cdf_inc = stats.norm.cdf(x_inc, shape, scale=scale)

        # scale cdf_inc and x-coordinates
        cdf_inc_scale = pdf[i] * cdf_inc / cdf_inc[-1]
        binw = edges[i+1] - edges[i]
        x_inc_scale = edges[i] + (binw * x_inc / x_inc[-1])

        # add to the new hd corodinates and cdf
        x_hd = np.append(x_hd, x_inc_scale)
        if i == 0:
            cdf_i  = 0
        else:
            cdf_i = cdf_hd[-1]
        cdf_hd = np.append(cdf_hd, cdf_inc_scale + cdf_i)

#        plt.plot(x_inc, cdf_inc)
#        plt.plot(x_inc_scale, cdf_inc_scale)

    cdf_new = interp(x_hd, cdf_hd, edges_new)
    # last point includes everything that comes after
    cdf_new[-1] = 1
    pdf_new = np.diff(cdf_new)
    widths_new = np.diff(edges_new)
    centers_new = widths_new + edges[0]
    # the first bin also includes everything that came before
    pdf_new[0] += cdf_new[0]
    pdf_new /= pdf_new.sum()

#    plt.plot(x_hd, cdf_hd)
#    plt.plot(edges_new, cdf_new, 'rs')
#
#    plt.bar(edges_new[:-1], pdf_new, width=widths_new, color='b')
#    plt.bar(edges[:-1], pdf, width=np.diff(edges), color='r', alpha=0.7)

    return centers_new, widths_new, pdf_new


def hist_centers2edges(centers):
    """Given the centers of bins, return its edges and bin widths.
    """

    binw_c = centers[1:] - centers[:-1]
    edges = np.ndarray((len(centers)+1,))
    edges[0] = centers[0] - binw_c[0]/2.0
    edges[-1] = centers[-1] + binw_c[-1]/2.0
    edges[1:-1] = centers[1:] - binw_c/2.0
    binw_e = edges[1:] - edges[:-1]

    return edges, binw_e


def df_dict_check_datatypes(df_dict):
    """
    there might be a mix of strings and numbers now, see if we can have
    the same data type throughout a column
    nasty hack: because of the unicode -> string conversion we might not
    overwrite the same key in the dict.
    """
    # FIXME: this approach will result in twice the memory useage though...
    # we can not pop/delete items from a dict while iterating over it
    df_dict2 = {}
    for colkey, col in df_dict.items():
        if len(col)==0:
            pass
        # if we have a list, convert to string
        elif type(col[0]).__name__ == 'list':
            for ii, item in enumerate(col):
                col[ii] = '*;*'.join(item)
        # if we already have an array (statistics) or a list of numbers
        # do not try to cast into another data type, because downcasting
        # in that case will not raise any exception
        elif type(col[0]).__name__[:3] in ['flo', 'int', 'nda']:
            df_dict2[str(colkey)] = np.array(col)
            continue
        # in case we have unicodes instead of strings, we need to convert
        # to strings otherwise the saved .h5 file will have pickled elements
        try:
            df_dict2[str(colkey)] = np.array(col, dtype=np.int32)
        except OverflowError:
            try:
                df_dict2[str(colkey)] = np.array(col, dtype=np.int64)
            except OverflowError:
                df_dict2[str(colkey)] = np.array(col, dtype=np.float64)
        except ValueError:
            try:
                df_dict2[str(colkey)] = np.array(col, dtype=np.float64)
            except ValueError:
                df_dict2[str(colkey)] = np.array(col, dtype=np.str)
        except TypeError:
            # in all other cases, make sure we have converted them to
            # strings and NOT unicode
            df_dict2[str(colkey)] = np.array(col, dtype=np.str)
        except Exception as e:
            print('failed to convert column %s to single data type' % colkey)
            raise(e)
    return df_dict2


def dict2df(df_dict, fname, save=True, update=False, csv=False, colsort=None,
            check_datatypes=False, rowsort=None, csv_index=False, xlsx=False,
            complib='blosc'):
        """
        Convert the df_dict to df and save/update if required. If converting
        to df fails, pickle the object. Optionally save as csv too.

        Parameters
        ----------

        df_dict : dict
            Dictionary that will be converted to a DataFrame

        fname : str
            File name excluding the extension. .pkl, .h5 and/or .csv will be
            added.
        """
        if check_datatypes:
            df_dict = df_dict_check_datatypes(df_dict)

        # in case converting to dataframe fails, fall back
        try:
            if colsort is not None:
                dfs = pd.DataFrame(df_dict)[colsort]
#                try:
#                    dfs = dfs[colsort]
#                except KeyError as e:
#                    print('Re-ordering the columns failed. colsort cols are:')
#                    print(sorted(colsort))
#                    print('Actual columns:')
#                    print(sorted(dfs.keys()))
#                    print('&', set(colsort) & set(dfs.keys()))
#                    print('-', set(colsort) - set(dfs.keys()))
#                    raise e
            else:
                dfs = pd.DataFrame(df_dict)
        except Exception as e:
            print('failed to convert to data frame', end='')
            if fname is not None:
                with open(fname + '.pkl', 'wb') as f:
                    pickle.dump(df_dict, f, protocol=2)
                print(', saved as dict')
            else:
                print('')
            print('df_dict has following columns and corresponding nr of rows')
            check_df_dict(df_dict)
            raise(e)

        if rowsort is not None:
            dfs.sort(columns=rowsort, inplace=True)

#        # apply categoricals to objects: reduce in memory footprint. In theory
#        # when using compression on a saved hdf5 object, this shouldn't make
#        # any difference.
#        for column_name, column_dtype in dfs.dtypes.iteritems():
#            # applying categoricals mostly makes sense for objects
#            # we ignore all others
#            if column_dtype.name == 'object':
#                dfs[column_name] = dfs[column_name].astype('category')

        # and save/update the statistics database
        if save and fname is not None:
            if update:
                print('updating: %s ...' % (fname), end='')
                try:
                    dfs.to_hdf('%s.h5' % fname, 'table', mode='r+', append=True,
                               format='table', complevel=9, complib=complib)
                except IOError:
                    print('Can not update, file does not exist. Saving instead'
                          '...', end='')
                    dfs.to_hdf('%s.h5' % fname, 'table', mode='w',
                               format='table', complevel=9, complib=complib)
            else:
                print('saving: %s ...' % (fname), end='')
                if csv:
                    dfs.to_csv('%s.csv' % fname, index=csv_index)
                if xlsx:
                    dfs.to_excel('%s.xlsx' % fname, index=csv_index)
                dfs.to_hdf('%s.h5' % fname, 'table', mode='w',
                           format='table', complevel=9, complib=complib)

            print('DONE!!\n')

        return dfs


class Tests(unittest.TestCase):

    def setUp(self):
        pass

    def test_sanitize_wine_prefix(self):

        wine_prefix = sanitize_wine_prefix('.wine32')
        self.assertEqual(wine_prefix, '$HOME/.wine32')

        wine_prefix = sanitize_wine_prefix('~/.wine32')
        self.assertEqual(wine_prefix, '$HOME/.wine32')

        wine_prefix = sanitize_wine_prefix('wine32')
        self.assertEqual(wine_prefix, '$HOME/wine32')

        notok = ['../wine', '/root/wine32', 'mnt/mimer/wine32', '.wine32/qq']
        for wine_prefix in notok:
            with self.assertRaises(ValueError):
                sanitize_wine_prefix(wine_prefix)

    def test_path_sanitize(self):

        paths = ['./not/allowed/',
                 '$not/$allowed/',
                 ' forget/about/it',
                 'forget/about/it ',
                 '!are\you\n\\NUTS??',
                 'don"t/dothis',
                 'don\'t',
                 'oh_look/a/ space'
                 '/not/../at/all',
                 '/not/at/all../',
                 'you/think\\youneed\\backspaces',
                 '../../for/db/dirs/this/is/notsafe',
                 'auch/aa/wild/c*rd',
                 'no/weird/-sub/dirs',
                 'no/weird/sub-/dirs',
                 'no/weird/sub./dirs',
                 'no/weird/-sub-/dirs',
                 '..nope/either8',
                 '&',
                 ]
        for path in paths:
            with self.assertRaises(ValueError):
                path_sanitize(path)
            with self.assertRaises(ValueError):
                path_sanitize('/' + path)
            with self.assertRaises(ValueError):
                path_sanitize(path, allowdd=True)
            with self.assertRaises(ValueError):
                path_sanitize(path, allowabs=True)

        # with allowdd active
        pathsok = ['../for/db/dirs/this/is/safe',
                   'what/aa/nice/path/',
                   'whatanicepath',
                   'ok/p-_/u_/in/this/case',
                   'ev.en.this.__is/oo/kk---000/',
                   ]
        for path in pathsok:
            path_sanitize(path, allowdd=True)
            with self.assertRaises(ValueError):
                path_sanitize('/' + path, allowdd=True)

        # with allowabs active
        pathsok = ['/for/db/dirs/this/is/safe',
                   '/what/aa/nice/path/',
                   '/whatanicepath',
                   '/ok/p-_/u_/in/this/case',
                   '/ev.en.this.__is/oo/kk---000/',
                   ]
        for path in pathsok:
            path_sanitize(path, allowabs=True)
            with self.assertRaises(ValueError):
                path_sanitize('..' + path, allowabs=True)

        with self.assertRaises(ValueError):
            path_sanitize('../../for/db/dirs/this/is/notsafe', allowdd=True)

        with self.assertRaises(ValueError):
            path_sanitize('./for/db/dirs/this/is/notsafe', allowabs=True)

    def test_rebin1(self):
        hist = np.array([2,5,5,9,2,6])
        bins = np.arange(7)
        nrbins = 3
        hist_, bins_ = rebin(hist, bins, nrbins)
        answer = np.array([7, 14, 8])
        self.assertTrue(np.allclose(answer, hist_))
        binanswer = np.array([0.0, 2.0, 4.0, 6.0])
        self.assertTrue(np.allclose(binanswer, bins_))

    def test_rebin2(self):
        hist = np.array([2,5,5,9,2,6])
        bins = np.arange(7)
        nrbins = 1
        hist_, bins_ = rebin(hist, bins, nrbins)
        answer = np.array([hist.sum()])
        self.assertTrue(np.allclose(answer, hist_))
        binanswer = np.array([bins[0],bins[-1]])
        self.assertTrue(np.allclose(binanswer, bins_))

    def test_rebin3(self):
        hist = np.array([1,1,1])
        bins = np.arange(4)
        nrbins = 2
        hist_, bins_ = rebin(hist, bins, nrbins)
        answer = np.array([1.5, 1.5])
        self.assertTrue(np.allclose(answer, hist_))
        binanswer = np.array([0, 1.5, 3.0])
        self.assertTrue(np.allclose(binanswer, bins_))

    def test_rebin4(self):
        hist = np.array([1,1,1])
        bins = np.arange(2, 14, 3)
        nrbins = 2
        hist_, bins_ = rebin(hist, bins, nrbins)
        answer = np.array([1.5, 1.5])
        self.assertTrue(np.allclose(answer, hist_))
        binanswer = np.array([2, 6.5, 11.0])
        self.assertTrue(np.allclose(binanswer, bins_))

    def test_rebin5(self):
        hist = np.array([1,4,2,5,6,11,9,10,8,0.5])
        bins = np.linspace(-2, 10, 11)
        nrbins = 8
        hist_, bins_ = rebin(hist, bins, nrbins)
        answer = np.array([2, 4, 4.75, 7.25, 13.25, 11.75, 11, 2.5])
        self.assertTrue(np.allclose(answer, hist_))
        binanswer = np.array([-2., -0.5, 1., 2.5, 4., 5.5, 7., 8.5, 10.0])
        self.assertTrue(np.allclose(binanswer, bins_))


if __name__ == '__main__':
    unittest.main()
