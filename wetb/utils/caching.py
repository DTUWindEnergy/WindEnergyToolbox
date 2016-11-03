'''
Created on 07/02/2014

@author: MMPE
'''
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
import sys
standard_library.install_aliases()
import inspect

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

