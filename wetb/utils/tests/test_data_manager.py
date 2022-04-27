
# %%
from tests import npt
import os
import pytest
from wetb.utils.cluster_tools.data_manager import data_manager


def test_zenodo():
    # Downloads source files and asserts equality on cheksums from metadata
    dm = data_manager(local=r'./tmp/',
                      source=r'10.5072/zenodo.77519',
                      platform='Zenodo',
                      platform_options={'sandbox': True})
    # Check if both test files are complete with md5 checksum
    for file in dm.metadata['files']:
        file_path = f"{dm.local}{file['key']}"
        npt.assert_equal(file['checksum'].split(':')[-1], dm._md5(file_path))
        os.remove(file_path)
    if len(os.listdir(path=dm.local)) == 0:
        os.rmdir(dm.local)

# %%
