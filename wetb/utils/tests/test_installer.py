import pytest

from wetb.utils.installer import install_wind_tool
import shutil

DESTINATION="/tmp/hawc2"

def test_installer_zip():
    # Install a program distributed as a zip file
    try:
        install_wind_tool(
            tool="HAWC2",
            destination=DESTINATION
        )
    except:
        shutil.rmtree(DESTINATION)
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
        shutil.rmtree(DESTINATION)
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
        shutil.rmtree(DESTINATION)
        raise
    finally:
        shutil.rmtree(DESTINATION)