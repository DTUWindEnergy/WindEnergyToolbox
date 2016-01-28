from six import exec_
import time
import inspect
def get_time(f):
    """Get time decorator
    returns (return_values, time_of_execution)

    >>> @get_time
    >>> def test():
    >>>    time.sleep(1)
    >>>    return "end"
    >>>
    >>> test()
    ('end', 0.999833492421551)
    """
    def wrap(*args, **kwargs):
        t = time.time()
        res = f(*args, **kwargs)
        return res, time.time() - t
    w = wrap
    w.__name__ = f.__name__
    return w


def print_time(f):
    """Print time decorator
    prints name of method and time of execution

    >>> @print_time
    >>> def test():
    >>>    time.sleep(1)
    >>>
    >>> test()
    test            1.000s
    """
    def wrap(*args, **kwargs):
        t = time.time()
        res = f(*args, **kwargs)
        print ("%-12s\t%.3fs" % (f.__name__, time.time() - t))
        return res
    w = wrap
    w.__name__ = f.__name__
    return w


cum_time = {}
def print_cum_time(f):
    """Print cumulated time decorator
    prints name of method and cumulated time of execution

    >>> @print_cum_time
    >>> def test():
    >>>    time.sleep(1)
    >>>
    >>> test()
    test            0001 calls, 1.000000s, 1.000000s pr. call'
    >>> test()
    test            0002 calls, 2.000000s, 1.000000s pr. call'
    """
    if f not in cum_time:
        cum_time[f] = (0, 0)

    def wrap(*args, **kwargs):
        t = time.time()
        res = f(*args, **kwargs)
        ct = cum_time[f][1] + time.time() - t
        n = cum_time[f][0] + 1
        cum_time[f] = (n, ct)
        print ("%-12s\t%.4d calls, %03fs, %fs pr. call'" % (f.__name__, n, ct, ct / n))
        return res
    w = wrap
    w.__name__ = f.__name__
    return w

def print_line_time(f):
    """Execute one line at the time and prints the time of execution.
    Only for non-branching and non-looping code

    prints: time_of_line, cumulated_time, code_line


    >>> @print_line_time
    >>> def test():
    >>>    time.sleep(.3)
    >>>    time.sleep(.5)
    >>>
    >>> test()
    0.300s    0.300s    time.sleep(.3)
    0.510s    0.810s    time.sleep(.51)

    """
    def wrap(*args, **kwargs):
        arg_names, varargs, varkw, defaults = inspect.getargspec(f)
        kwargs[varargs] = args[len(arg_names):]
        kwargs[varkw] = {}
        for k, v in kwargs.items():
            if k not in tuple(arg_names) + (varargs, varkw):
                kwargs.pop(k)
                kwargs[varkw][k] = v
        if defaults:
            kwargs.update(dict(zip(arg_names[::-1], defaults[::-1])))
        kwargs.update(dict(zip(arg_names, args)))


        lines = inspect.getsourcelines(f)[0][2:]
        tcum = time.time()
        locals = kwargs
        gl = f.__globals__

        for l in lines:
            tline = time.time()
            exec(l.strip(), locals, gl)  #res = f(*args, **kwargs)
            print ("%.3fs\t%.3fs\t%s" % (time.time() - tline, time.time() - tcum, l.strip()))
    w = wrap
    w.__name__ = f.__name__
    return w
