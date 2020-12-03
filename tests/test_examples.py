import importlib
import os
import pkgutil
import warnings
import mock
import pytest
import matplotlib.pyplot as plt
import sys
from examples import examples
from tests.run_main import run_module_main


def get_main_modules():
    package = examples
    modules = []
    for _, modname, _ in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = importlib.import_module(modname)

        if 'main' in dir(m):
            modules.append(m)
    return modules


def print_main_modules():
    print("\n".join([m.__name__ for m in get_main_modules()]))


@pytest.mark.parametrize("module", get_main_modules())
def test_main(module):
    run_module_main(module)


if __name__ == '__main__':
    print_main_modules()
