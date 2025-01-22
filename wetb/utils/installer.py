import glob
import io
import json
import os
import platform
import shutil
import sys
import zipfile
from pathlib import Path
from platform import architecture
from urllib.request import Request
from urllib import request
import ssl
import certifi


def urlopen(*args, **kwargs):
    return request.urlopen(
        *args, **kwargs, context=ssl.create_default_context(cafile=certifi.where())
    )


def chmod_x(exe_path: str):
    """Utility function to change the file mode of a file to allow execution

    Parameters
    ----------
    path : str
        path to the file to apply the +x policy to.
    """
    st = os.stat(exe_path)
    os.chmod(exe_path, st.st_mode | 0o111)


def install_wind_tool(
    tool: str = None, version: str = None, platform: str = None, destination: str = None
):
    """Utility function to download and install any tool available from the HAWC2 tools website

    Parameters
    ----------
    tool : str, required
        The tool to install. If None, the function will return immediately without doing anything. By default None
    version : str, optional
        Version of the software. If None, the latest version will be downloaded. By default None
    platform : str, optional
        The build platform of the tool. If None, the platform of the executing machine will be used. By default None
    destination : str, optional
        Destination path for the download / installation. If None, the destination is set to cwd. By default None
    """

    # Escape backslash if windowspath is given
    destination = Path(destination.encode("unicode_escape").decode()).as_posix()

    if tool is None:
        print("No tool has been given for install. Nothing has been installed.")
        return

    # If platform is not set, get the platform of the executing machine
    if platform is None:
        if os.name == "nt":
            if architecture()[0][:2] == "32":
                platform = "win32"
            else:
                platform = "win64"
        else:
            platform = "linux"
    else:
        platform = platform.lower()

    # Check if tool is available on the tools website
    req = Request("https://tools.windenergy.dtu.dk/version_inventory.json")
    versions = json.loads(urlopen(req).read())
    if tool not in versions:
        print(
            f"The tool '{tool}' is not available in the inventory on tools.windenergy.dtu.dk."
        )
        return

    # Check if requested version is available, and default it is not.
    if version is not None and version not in versions[tool]["available"]:
        print(
            f"Version '{version}' of '{tool}' is not available - defaulting to the latest version: '{versions[tool]['latest']}'"
        )
        version = versions[tool]["latest"]
    elif version is None:
        version = versions[tool]["latest"]
        print(f"Using latest version of '{tool}': {version}")

    # Get a list of the different types of executables available for the given platform
    req = Request("https://tools.windenergy.dtu.dk/product_inventory.json")
    download = json.loads(urlopen(req).read())
    exes = download[tool][version][platform]
    types = [exes[app]["type"] for app in exes]

    # Select the standalone executable if that is available, otherwise select the zip archive for download
    for ind, file in enumerate(exes):
        if "executable" in types[ind].lower() and "standalone" in types[ind].lower():
            req = Request(f"https://tools.windenergy.dtu.dk{exes[file]['link']}")
            break
        elif file.split(".")[-1].lower() == "zip":
            req = Request(f"https://tools.windenergy.dtu.dk{exes[file]['link']}")
            break

    # Download the file to a buffer object
    buffer = io.BytesIO(urlopen(req).read())

    # If destination is not provided, set the installation destination to cwd.
    if destination is None:
        destination = os.getcwd()

    print(f"Installing {tool} version {version} for {platform}")
    os.makedirs(destination, exist_ok=True)
    # If the download is a zip archive, extract the archive to the destination directory
    if req.full_url.endswith(".zip"):
        zipfile.ZipFile(buffer).extractall(path=destination)
    # If the download is not a zip archive, but an executable, save the bytes object to a file
    else:
        with open(f"{destination}/{req.full_url.split('/')[-1]}", "wb") as f:
            f.write(buffer.getvalue())

    # Add execution policy to the executable files (files with .exe or no extension)
    for file in glob.glob(f"{destination}/*"):
        if file.endswith(".exe") or os.path.splitext(file)[-1] == "":
            chmod_x(file)

    print(f"{tool} version {version} succesfully installed in {destination}")

    return


def install_hawc2_dtu_license():
    """Function to install the DTU HAWC2 license. In order to install the license, you must be logged in to the DTU network."""
    install_dtu_license("hawc2")


def install_hawcstab2_dtu_license():
    """Function to install the DTU HAWCStab2 license. In order to install the license, you must be logged in to the DTU network."""
    install_dtu_license("hawcstab2")


def install_ellipsys_dtu_license():
    """Function to install the DTU HAWCStab2 license. In order to install the license, you must be logged in to the DTU network."""
    install_dtu_license("ellipsys")


def install_dtu_license(software : str):
    """Function to install the DTU online license for HAWC2, HAWCStab2 and Ellipsys. In order to install the license, you must be logged in to the DTU network."""
    software = software.lower()
    assert software in ["hawc2", "hawcstab2", "ellipsys"], "Argument 'software' must be one of ['hawc2', 'hawcstab2,' 'ellipsys']"
    if sys.platform.lower() == "win32":
        f = Path(os.getenv("APPDATA")) / f"DTU Wind Energy/{software}/license.cfg"
    else:
        f = Path.home() / f".config/{software}/license.cfg"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        r = urlopen("http://license-internal.windenergy.dtu.dk:34523").read()
        if b"LICENSE SERVER RUNNING" in r:
            f.write_text(
                "[licensing]\nhost = http://license-internal.windenergy.dtu.dk\nport = 34523"
            )
        else:
            raise ConnectionError(
                f"Could not connect to the DTU license server. You must be connected to the DTU network to use this function."
            )




def install_keygen_license(software: str, cfg_file: str, force: bool = False):
    """Install license file for HAWC2, HAWCStab2 or Ellipsys on your machine

    Parameters
    ----------
    software : str
        Must be one of HAWC2, HAWCStab2 or Ellipsys. The argument is case insensitive.
    cfg_file : str
        Path to the license file to install
    force : bool, optional
        Switch to force the installation, overwriting any existing , by default False

    Returns
    -------
    NoneType
        None

    Raises
    ------
    ValueError
        A ValueError is raised if the name of the software argument is not supported.
    """

    SUPPORTED_SOFTWARES = ["hawc2", "hawcstab2", "ellipsys"]
    if software.lower() not in SUPPORTED_SOFTWARES:
        raise ValueError(f"'software' must be one of {SUPPORTED_SOFTWARES}")

    USER_PLATFORM = platform.uname().system

    if USER_PLATFORM == "Windows":
        APPDATA = f"{os.environ['APPDATA']}"
    else:
        APPDATA = "None"

    def local_license_dir(platform, software):
        return {
            "Windows": os.path.join(
                APPDATA,
                "DTU Wind Energy",
                f"{software}",
            ),
            "Linux": os.path.join(f"{Path.home()}", ".config", f"{software}"),
            "Darwin": os.path.join(f"{Path.home()}", "Library", "Application Support"),
        }[platform]

    def local_license_file(software):
        return {
            "hawc2": "license.cfg",
            "hawcstab2": "license.cfg",
            "pywasp": "",
            "ellipsys": "license.cfg",
        }[software.lower()]

    license_path = local_license_dir(USER_PLATFORM, software)
    lic_name = local_license_file(software)

    os.makedirs(license_path, exist_ok=True)

    if (
        os.path.exists(os.path.join(license_path, lic_name))
        and os.path.isfile(os.path.join(license_path, lic_name))
        and (not force)
    ):
        print(
            f"License already installed for {software}, use 'force=True' to overwrite installation"
        )
    else:
        shutil.copy(f"{cfg_file}", f"{os.path.join(license_path,lic_name)}")
