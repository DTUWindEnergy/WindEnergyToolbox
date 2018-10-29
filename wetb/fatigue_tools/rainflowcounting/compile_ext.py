
if __name__ == '__main__':
    from distutils.core import setup
    from distutils.extension import Extension
    from Cython.Distutils import build_ext
    import numpy
    import os

    path = os.path.dirname(__file__)
    ext_modules = [Extension("rainflowcount_IEA", [path + "/rainflowcount_IEA.pyx"], include_dirs=[numpy.get_include()]),
                   Extension("rainflowcount_astm", [path + "/rainflowcount_astm.pyx"],
                             include_dirs=[numpy.get_include()])
                   ]

    setup(
        name='name',
        cmdclass={'build_ext': build_ext},
        package_dir={'': path},
        ext_modules=ext_modules)
