import pytest

from wetb.utils.installer import install_wind_tool


def test_installer_zip():
    # Install a program distributed as a zip file
    install_wind_tool(
        tool="HAWC2",
        destination="/tmp/hawc2"
    )


def test_installer_executable():
    # Install a program distributed as a standalone executable
    install_wind_tool(
        tool="license_manager",
        destination="/tmp/hawc2"
    )