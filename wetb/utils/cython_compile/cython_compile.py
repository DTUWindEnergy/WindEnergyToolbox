'''
Created on 10/07/2013

@author: Mads M. Pedersen (mmpe@dtu.dk)


Wrapper functions and decorators for compiling functions using Cython

'''
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import dict
from builtins import open
from builtins import zip
from builtins import str
from future import standard_library
standard_library.install_aliases()
import inspect
import os
import re
import shutil
import subprocess
import sys
import numpy as np
import warnings
import imp



def wrap(f, autodeclare, *args, **kwargs):
    """
    Wrapper function returned by the cython_compile and cython_compile_autodeclare decorators

    :param f: Function to compile
    :type f: function
    :param py2pyx_func: py2pyx or py2pyx_autodeclare
    :type py2pyx_func: function
    """

    # Generate name: "c:\documents\project\mymodule.py" -> mymodule_myfunction
    # When called from ipython notebooks, filename is an object e.g: "<ipython-input-12-e897f9fefc0c>"
    # therefore <,>,- is removed to make it a legal python module name
    name = os.path.relpath(inspect.getabsfile(f), str(os.getcwd())).replace(".py", "")
    name = name.replace("<", "").replace(">", "").replace("-", "")
    name = "%s_%s" % (name, f.__name__)
    if name.startswith(".."):
        rel_dir = os.path.dirname(name) + os.path.sep
        sys.path.append(rel_dir)
        name = name[len(rel_dir):]
        pass

    module_name = name.replace(os.path.sep, ".")

    # import compiled module_name if exists, otherwise compile and import
    try:
        cmodule = __import__(module_name, fromlist="*")
        if cmodule.__name__ != module_name or not hasattr(cmodule, f.__name__):
            raise ImportError()
    except ImportError:
        # Generate pyrex code lines
        if autodeclare:
            pyx_lines = py2pyx_autodeclare(f, args, kwargs.copy())
        else:
            # source lines except '@cython_compile'
            source_lines = inspect.getsourcelines(f)[0][1:]
            indent = source_lines[0].index("def")
            pyx_lines = py2pyx([l[indent:] for l in source_lines])

        # Write pyrex code lines to .pyx file
        pyx_filename = name.replace("\\", "/") + ".pyx"
        with open(pyx_filename, 'w', encoding='utf-8') as fid:
            fid.writelines(pyx_lines)

        # compile, import compiled module_name and delete temporary files
        cmodule = compile_and_cleanup(module_name, pyx_filename, module_name)
    try:
        cf = getattr(cmodule, f.__name__)
        if kwargs == {}:
            return cf(*args)
        else:
            return cf(*args, **kwargs)
    except AttributeError:
        warnings.warn(
            "Compilation or import of %s failed. Python function used instead" %
            f)
        return f(*args, **kwargs)


def cython_compile(f):
    """Decorator for compilation, import and execution of the function, f.
    Variables can be declared using Pure or cdef syntax, see module description
    Example:
    @cython_compile
    def my_func(p):
        pass
    """
    w = lambda *args, **kwargs: wrap(f, False, *args, **kwargs)
    w.__name__ = f.__name__
    return w


def cython_compile_autodeclare(f):
    """Decorator for autodeclaring, compilation, import and execution of the function, f.
    Declared variables using cdef syntax overrides autodeclaration, see module description
    Example:
    @cython_compile_autocompile
    def my_func(p):
        pass
    """
    w = lambda *args, **kwargs: wrap(f, True, *args, **kwargs)
    w.__name__ = f.__name__
    return w


def cython_import(import_module_name, compiler=None):
    """Compiles and imports a module. Use it similar to the normal import
    Example (import my_func from my_module):

    from wetb.utils.cython_compile import cython_import
    cython_import('my_module')
    import my_module # import must be after cython_import statement
    my_module.my_func()

    """
    module_name = import_module_name.split(".")[-1]
    globals()[module_name] = __import__(import_module_name, fromlist=module_name)

    pyd_module = eval(module_name)
    if not is_compiled(pyd_module):

        # Read py-module
        file_name = import_module_name.replace(".", "/") + ".py"
        for p in ["."] + sys.path:
            if os.path.isfile(os.path.join(p, file_name)):
                file_path = os.path.join(p, file_name).replace("\\", "/")
                break
        else:
            raise IOError("Module %s cannot be found" % file_name)

        fid = open(file_path)
        pylines = fid.readlines()
        fid.close()

        # write pyrex file
        pyx_filename = file_path.replace('.py', '.pyx')
        fid = open(pyx_filename, 'w')
        pyxlines = py2pyx(pylines)
        fid.writelines(pyxlines)
        fid.close()

        # compile, import compiled module and delete temporary files
        module_relname = os.path.relpath(eval(module_name).__file__, str(os.getcwd())).replace(os.path.sep, ".")[:-3]
        return compile_and_cleanup(import_module_name, pyx_filename, module_relname, compiler)
    return eval(module_name)


def compile_and_cleanup(module, pyx_filename, module_relname, compiler=None):
    """compile, import compiled module and delete temporary files"""

    # Generate setup.py script
    fid = open('setup.py', 'w')
    setup_str = """from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("%s", ["%s"], include_dirs = [numpy.get_include()])]

setup(
name = 'name',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)""" % (module_relname, pyx_filename)
    fid.write(setup_str)
    fid.close()
    if hasattr(sys, 'frozen'):
        raise Exception ("Trying to compile %s in frozen application" % str(module))
    # create compile command
    import platform
    if compiler is not None:
        compiler_str = "--compiler=%s" % compiler
    elif platform.architecture()[0] == "32bit" and os.name == 'nt' and "mingw" in os.environ['path'].lower():
        compiler_str = "--compiler=mingw32"
    else:
        compiler_str = ""



    bin_python = sys.executable
    cmd = "%s setup.py build_ext --inplace %s" % (bin_python, compiler_str)

    # compile
    print ("compiling %s: %s" % (module, cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    (out, err) = proc.communicate()

    # Reload and check that module is compiled
    try:
        try:
            del sys.modules[module]
        except KeyError:
            pass
        cmodule = __import__(module, fromlist=module.split(".")[-1])
        imp.reload(cmodule)

    except ImportError:
        cmodule = None

    if cmodule is None or is_compiled(cmodule) == False:
        line = '\n' + '=' * 79 + '\n'
        sys.stderr.write("%s was not compiled correctly. It may result in slower execution" % module)
        sys.stderr.write('%sstdout:\n%s' % (line, out.decode()))
        sys.stderr.write('%sstderr:%s%s' % (line, line, err))

    else:
        print ("Compiling succeeded")
        for f in ['setup.py', pyx_filename, pyx_filename.replace(".pyx", '.c')]:
            if os.path.isfile(f):
                os.remove(f)

    # Clean up. Remove temporary files and folders
    if os.path.isdir("build"):
        try:
            shutil.rmtree("build")
        except:
            pass

    return cmodule


def py2pyx_autodeclare(f, args, kwargs):
    """Generate pyrex code of function f and its input arguments
    This function invokes py2pyx and extends with autodeclarations:
    - arguments: input arguments are declared based their values in args and kwargs
    - cdef: Variables declared by cdef, overrides autodeclaration, e.g:
        "def func(a): #cpdef func(int a):" -> "cpdef func(int a):" (independent of type of a)
        "#cdef int a" - "cdef int a" (independent of type of a)
    - assignment: Variables assigned in function, e.g.
        "a = xxx"
      are declared based on the type of eval(xxx)
    - in: variables returned by iterators, e.g:
        "for a in range(5)",
        "for a,b in [(1,.2)]:"
      are declared base on the type of the first element in eval(iterator), e.g. "type(eval(range(5)[0]))"
    """

    arg_names, _, _, defaults = inspect.getargspec(f)

    # update kwargs with defaults
    if defaults:
        for k, v in zip(arg_names[::-1], defaults[::-1]):
            if k not in kwargs:
                kwargs[k] = v
    kwargs.update(dict(zip(arg_names, args)))

    # get pyx code lines using py2pyx
    lines = inspect.getsourcelines(f)[0]
    if lines[0].strip() == "@cython_compile_autodeclare":
        lines = lines[1:]
    lines = py2pyx(lines)

    # prepare regular expressions
    var_name = "(?:_*[a-zA-Z][a-zA-Z0-9_]*)"  # optional "_" + one alpha + [0..n] x alphanum. "?:" = no group
    reg_args = re.compile("[ \t]*def +(?:%s) *\(([^:]*)\) *:" % (var_name))
    reg_import = re.compile("[ \t]*c?import *(%s)(?: +as +(%s))?" % (var_name, var_name))
    reg_cdef = re.compile("[ \t]*cdef +(?:(?:signed|unsigned|int|long|float|double|np\.ndarray\[.*\]) *)*(%s)" % var_name)
    reg_assign = re.compile('[ \t]*(%s) *=(.*)' % var_name)
    reg_in = re.compile('[ \t]*for +(%s(?:, *%s)*) +in +(.*):' % (var_name, var_name))


    def declare_str(var_name, var_value):
        """Generate declaration string '<type(var_value)> <var_name>' e.g:
        declare_str('a',1) -> "long a"
        """
        if isinstance(var_value, (int)):
            return "long long %s" % var_name
        if isinstance(var_value, float):
            return "double %s" % var_name
        elif isinstance(var_value, np.ndarray):
            return "np.ndarray[np.%s_t,ndim=%d] %s" % (var_value.dtype, len(var_value.shape), var_name)
        else:
            raise NotImplementedError(type(var_value))

    defs = {}  # dict for known local variables
    def_line = None  # line nr of "def func():". Autodeclaration of local field inserted below this line
    for i, line in enumerate(lines):
        if def_line is None and 'def' in line:
            # first line containing "def" = function declaration line
            def_line = i
            match = reg_args.match(line)
            if match is not None:
                args = match.group(1).strip()  # line="   def func(xxx):#comment" -> args='xxx'
                arg_strs = []
                if args != "":
                    for arg in args.split(','):
                        arg_name = arg.split('=')[0].strip()
                        arg_value = kwargs[arg_name]
                        try:
                            arg_strs.append(arg.strip().replace(arg_name, declare_str(arg_name, arg_value), 1))
                        except NotImplementedError:
                            arg_strs.append(arg)
                # replace function declaration line
                lines[i] = '%scpdef %s(%s):\n' % (" " * line.index('def'), f.__name__, ", ".join(arg_strs))
        else:
            #Import: import x as y
            match = reg_import.match(line)
            if match is not None:
                # add imported moduled to kwargs -> enable evaluation of variables
                import_module = match.group(1)
                if match.group(2) is None:
                    kwargs[import_module] = __import__(import_module)
                else:
                    # import xxx as y
                    kwargs[match.group(2)] = __import__(import_module)
                continue
            #Cdef: cdef int x
            match = reg_cdef.match(line)
            if match is not None:
                # line contains a 'cdef' declaration.
                # Add to defs to avoid redeclaration
                var_name = match.group(1)
                if var_name not in defs:
                    defs[var_name] = None
                continue
            #Assignment: x = y
            match = reg_assign.match(line)
            if match is not None:
                # line contains an assignment, e.g. a = xxx.
                # Try to evaluate xxx and declare a as type of eval(xxx)
                try:
                    var_name = match.group(1)
                    if var_name not in defs:
                        var_value = eval(match.group(2), globals(), kwargs)
                        defs[var_name] = declare_str(
                            var_name.strip(), var_value)
                        kwargs[var_name] = var_value
                except NotImplementedError:
                    pass
                continue
            #In: For x in y
            match = reg_in.match(line)
            if match is not None:
                # line contains 'for xxx in yyy:'.
                # Try to evaluate yyy and declare xxx as type of first element
                # of eval(yyy)
                var_names = [v.strip() for v in match.group(1).split(",")]
                var_values = eval(match.group(2), globals(), kwargs)[0]
                if not isinstance(var_values, (list, tuple)):
                    var_values = (var_values,)
                for var_name, var_value in zip(var_names, var_values):
                    try:
                        if var_name not in defs:
                            defs[var_name] = declare_str(var_name, var_value)
                            kwargs[var_name] = var_value
                    except NotImplementedError:
                        pass
    indent = lines[def_line + 1].replace(lines[def_line + 1].lstrip(), "")

    # Insert declaration of local fields ordered by name just below function
    # declaration
    for key in sorted(defs.keys(), reverse=True):
        if defs[key] is not None:
            lines.insert(def_line + 1, "%scdef %s\n" % (indent, defs[key]))

    return lines


def py2pyx(pylines):
    """Generate pyrex code lines from python code lines
    - Adds import of cython and numpy
    - searches every line for "#c". If found text before "#c" is replaced with text after "#c", e.g:
    "def func(a): #cpdef func(int a):" -> "cpdef func(int a):
    "#cdef int b" -> "cdef int b"
    """
    pyxlines = ['import cython\n', 'import numpy as np\n', 'cimport numpy as np\n']
    for l in pylines:

        if "#c" in l:
            indent = l[:len(l) - len(l.lstrip())]
            cdef = l[l.index("#c") + 1:]
            l = indent + cdef
        pyxlines.append(l)
    return pyxlines


def is_compiled(module):
    return module.__file__.lower()[-4:] == ".pyd" or module.__file__.lower()[-3:] == ".so"
