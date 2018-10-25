import os
import sys

from unittest import mock
import pytest
import matplotlib.pyplot as plt


def run_module_main(module):
    # check that all main module examples run without errors
    if os.name == 'posix' and "DISPLAY" not in os.environ:
        pytest.xfail("No display")

    def no_show(*args, **kwargs):
        pass
    plt.show = no_show  # disable plt show that requires the user to close the plot

    def no_print(s):
        pass

    try:
        with mock.patch.object(module, "__name__", "__main__"):
            with mock.patch.object(module, "print", no_print):
                getattr(module, 'main')()
    except Exception as e:
        raise type(e)(str(e) +
                      ' in %s.main' % module.__name__).with_traceback(sys.exc_info()[2])
