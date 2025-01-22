import platform
import pathlib
import os


import pytest

from wetb.utils.installer import install_wind_tool, install_hawc2_dtu_license, install_keygen_license, install_hawcstab2_dtu_license, install_ellipsys_dtu_license
import shutil

DESTINATION="/tmp/hawc2"

USER_PLATFORM = platform.uname().system
if USER_PLATFORM == "Windows":
    APPDATA = f"{os.environ['APPDATA']}"
else:
    APPDATA = "None"


TEST_LICENSE_FILE = "/tmp/license.cfg"
with open(TEST_LICENSE_FILE, "w") as file:
    file.writelines(["\n[licensing]", "\nhost: www.fakehost.com", "\nport=360"])
    

def local_license_dir(platform, software):
    return {
        "Windows": os.path.join(
            APPDATA,
            "DTU Wind Energy",
            f"{software}",
        ),
        "Linux": os.path.join(f"{pathlib.Path.home()}", ".config", f"{software}"),
        "Darwin": os.path.join(
            f"{pathlib.Path.home()}", "Library", "Application Support"
        ),
    }[platform]

def local_license_file(software):
    return {
        "hawc2": "license.cfg",
        "hawcstab2": "license.cfg",
        "pywasp": "",
        "ellipsys": "license.cfg",
    }[software.lower()]


def test_installer_zip():
    # Install a program distributed as a zip file
    try:
        install_wind_tool(
            tool="HAWC2",
            destination=DESTINATION
        )
    except:
        raise
    finally:
        shutil.rmtree(DESTINATION)

def test_installer_executable():
    # Install a program distributed as a standalone executable
    try:
        install_wind_tool(
            tool="license_manager",
            destination=DESTINATION
        )
    except:
        raise
    finally:
        shutil.rmtree(DESTINATION)

def test_version_not_available():
    # Test fallback to latest version
    try:
        install_wind_tool(
            tool="HAWC2",
            version="1.2.3",
            destination=DESTINATION
        )
    except:
        raise
    finally:
        shutil.rmtree(DESTINATION)


@pytest.mark.parametrize("software", ["HAwC2","HAWCStab2", "ellipsys"])
def test_install_dtu_license(software):
    software = software.lower()
    license_path = local_license_dir(USER_PLATFORM, "HAWC2")

    if software == "hawc2":
        install_hawc2_dtu_license()
    elif software == "hawcstab2":
        install_hawcstab2_dtu_license()
    elif software == "ellipsys":
        install_ellipsys_dtu_license()
        
    assert os.path.exists(f"{local_license_dir(USER_PLATFORM, software)}/{local_license_file(software)}")

    shutil.rmtree(license_path, ignore_errors=True)




TEST_LICENSE_FILE = "/tmp/license.cfg"
with open(TEST_LICENSE_FILE, "w") as file:
    file.writelines(["\n[licensing]", "\nhost: www.fakehost.com", "\nport=360"])
@pytest.mark.parametrize("software,license", [("HAwC2", TEST_LICENSE_FILE),("HAWCStab2", TEST_LICENSE_FILE),("ellipSYS", TEST_LICENSE_FILE)])
def test_install_keygen_license(software, license):
    license_path = local_license_dir(USER_PLATFORM, software)
    try:
        install_keygen_license(software=software, cfg_file=license)
        assert os.path.exists(f"{local_license_dir(USER_PLATFORM, software)}/{local_license_file(software)}")
    except Exception as exc:
        raise exc
    finally:
        shutil.rmtree(license_path, ignore_errors=True)