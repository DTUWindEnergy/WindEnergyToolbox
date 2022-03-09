'''
Created on 07/02/2014

@author: MMPE
'''
import sys
from collections import OrderedDict
import os
import inspect
import numpy as np
def set_cache_property(obj, name, get_func, set_func=None):
    """Create a cached property

    Parameters
    ----------
    obj : object
        Class to add property to
    name : str
        Name of property
    get_func : func
        Getter function
    set_func : func, optional
        Setter function

    Examples
    --------
    >>> class Example(object):
    >>>     def __init__(self):
    >>>        set_cache_property(self, "test", self.slow_function)
    >>>
    >>> e = Example()
    >>> e.test # Call, store and return result of e.slow_function
    >>> e.test # Return stored result of e.slow_function
    >>> e._test = None # clear cache result
    >>> e.test # Call, store and return result of e.slow_function
    """
    _name = "_" + name
    setattr(obj, _name, None)
    def get(self):
        if getattr(obj, _name) is None:
            setattr(obj, _name, get_func())
        return getattr(obj, _name)

    p = property(lambda self:get(self), set_func)
    return setattr(obj.__class__, name, p)

def cache_function(f):
    """Cache function decorator

    Example:
    >>> class Example(object):
    >>>    @cache_function
    >>>    def slow_function(self):
    >>>        # calculate slow result
    >>>        return 1
    >>>
    >>> e = Example()
    >>> e.slow_function() # Call, store and return result of e.slow_function
    >>> e.slow_function() # Return stored result of e.slow_function
    """
    def wrap(*args, **kwargs):
        self = args[0]
        name = "_" + f.__name__
        if not hasattr(self, name) or getattr(self, name) is None or kwargs.get("reload", False):
            try:
                del kwargs['reload']
            except KeyError:
                pass
            # ======HERE============
            setattr(self, name, f(*args, **kwargs))
            # ======================
            if not hasattr(self, "cache_attr_lst"):
                self.cache_attr_lst = set()
                def clear_cache():
                    for attr in self.cache_attr_lst:
                        delattr(self, attr)
                    self.cache_attr_lst = set()
                self.clear_cache = clear_cache
            self.cache_attr_lst.add(name)

        return getattr(self, name)
    version = sys.version_info
    if version >= (3,3):
        if 'reload' in inspect.signature(f).parameters.values():
            raise AttributeError("Functions decorated with cache_function are not allowed to take a parameter called 'reload'")
    elif 'reload' in inspect.getargspec(f)[0]:
        raise AttributeError("Functions decorated with cache_function are not allowed to take a parameter called 'reload'")
    return wrap

class cache_method():
    def __init__(self, N):
        self.N = N
        
        
    def __call__(self, f):
        def wrapped(caller_obj, *args):
            name = "_" + f.__name__
            arg_id = ";".join([str(a) for a in args])
            if not hasattr(caller_obj,'%s_cache_dict'%name):
                setattr(caller_obj,'%s_cache_dict'%name, OrderedDict())
            cache_dict = getattr(caller_obj,'%s_cache_dict'%name)
            if arg_id not in cache_dict: 
                cache_dict[arg_id] = f(caller_obj, *args)
                if len(cache_dict)>self.N:
                    cache_dict.popitem(last=False)
            return cache_dict[arg_id]
        return wrapped
    
def cache_npsave(f):
    def wrap(filename,*args,**kwargs):
        np_filename = os.path.splitext(filename)[0] + ".npy"
        def loadsave():
            res = f(filename,*args,**kwargs)
            np.save(np_filename,res)
            return res
        if os.path.isfile(np_filename) and (not os.path.isfile(filename) or os.path.getmtime(np_filename) > os.path.getmtime(filename)):
            try:
                return np.load(np_filename)
            except:
                return loadsave()
        else:
            return loadsave()
    return wrap

def _get_npsavez_wrap(f, compress):
    def wrap(filename,*args,**kwargs):
        np_filename = os.path.splitext(filename)[0] + ".npy%s.npz"%("",".c")[compress]
        def loadsave():
            res = f(filename,*args,**kwargs)
            if compress:
                np.savez_compressed(np_filename,*res)
            else:
                np.savez(np_filename,*res)
            return res
        if os.path.isfile(np_filename) and (not os.path.isfile(filename) or os.path.getmtime(np_filename) > os.path.getmtime(filename)):
            try:
                npzfile = np.load(np_filename)
                return [npzfile['arr_%d'%i] for i in range(len(npzfile.files))]
            except:
                return loadsave()
        else:
            return loadsave()
    return wrap

def cache_npsavez(f):
    return _get_npsavez_wrap(f,False)


def cache_npsavez_compressed(f):
    return _get_npsavez_wrap(f, True)