import io
import json
import os
import zipfile
from urllib.request import Request, urlopen


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
    if tool == None:
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
    if version != None and version not in versions[tool]["available"]:
        version = versions[tool]["latest"]
        print(
            f"Version '{version}' of '{tool}' is not available - defaulting to the latest version: '{version}'"
        )
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

    # If the download is a zip archive, extract the archive to the destination directory
    if req.full_url.endswith(".zip"):
        zipfile.ZipFile(buffer).extractall(path=destination)
    # If the download is not a zip archive, but an executable, save the bytes object to a file
    else:
        with open(f"{destination}/{req.full_url.split('/')[-1]}", "wb") as f:
            f.write(buffer.getvalue())

    print(f"{tool} version {version} succesfully installed in {destination}")

    return
