# %%
from abc import ABC
import os
from wetb.utils.cluster_tools.ssh_client import SSHClient
import getpass
from tqdm import tqdm
import socket


class DownloadProgressBar(tqdm):
    """Progressbar for download of data,
    for use with callbacks.
    """
    def update_to(self, current, total):
        self.total = total
        self.update(current - self.n)


class data_manager(ABC):
    """A Data manager to handle large files in shared projects.
    When the data manager is intialized it will check if the data file is present locally.
    If it is not it will be downloaded from a given online source.
    Currently only Sophia is implemented using the "wetb" SSHClient.

    With relatively little effort a different platform can easily be implemented.
    Such as Zenodo or DTU Data.
    """

    def __init__(self, local, source, platform, unzip=True, checksum=None, platform_options=None):
        """Initialize data manager, with source and destination

        Parameters
        ----------
        local : str
            Path to data destination on local machine.
        source : str
            Platform dependend location of data.
            URL, DOI or path.
        platform : str
            Online resource to download data from.
            Current available platforms:
                "Sophia"
        platform_options : dict
            Options for platform to function.
            For e.g. for Sophia "pkey" option.
        """
        # Store parameters
        self.local = local
        self.source = source
        self.platform = platform
        self.platform_options = platform_options
        self.checksum = checksum

        # TODO: Re-structure code with specific Sophia __call__ to remove reference to Sophia in __init__
        # Check if already on Sophia
        if 'hpc' in socket.gethostname():
            print(f"Running on Sophia no download required")
            data_path = self.source
        else:
            # Call main function
            data_path = self.__call__()
            if unzip:
                self._unzip()
            self._sha256sum()

        self.data_path = data_path

    def __call__(self):
        """Evaluate if data is prensent locally.
        or alternatively download it.
        """
        if os.path.exists(self.local):
            print(f"Data present in local directory: {self.local}")
        else:
            self._download()
            print(f"Data downloaded to local directory: {self.local}")
        data_path = self.local
        return data_path

    def _download(self):
        """Use the chosen platform to download the data file.
        """
        if self.platform == "Sophia":
            # Setup SSH Call
            user = getpass.getuser()
            SSHCall = {'username': user, 'port': 22, 'host': 'sophia.dtu.dk'}
            if "pkey" in self.platform_options:
                SSHCall['key'] = self.platform_options['pkey']
            else:
                password = getpass.getpass(prompt='Provide Sophia password: ')
                SSHCall['password'] = password

            client = SSHClient(**SSHCall)
            # Delete SSHCall as to not keep password in memory
            del SSHCall

            # Download using wetb SSH Client
            with DownloadProgressBar(unit='B', unit_scale=True) as t:
                client.download(self.source, self.local, verbose=False, callback=t.update_to)

            # Check file is not empty, as SSH CLient returns an empty file with correct name when failing.
            if os.stat(f"{destination_dir}/{filename}").st_size == 0:
                raise RuntimeError("Dowloaded file is empty, please check source is an existsing file.")

    def _unzip(self):
        """Unzip file if zipped.
        # TODO: Remake function to unpack multiple file formats e.g. .tar or .7zip
        """
        if self.local.split('.')[-1] == 'zip':
            from zipfile import ZipFile
            # Extract zipfile
            unzipped_folder_dir = self.local.split('.')[0]
            os.makedirs(unzipped_folder_dir)
            with ZipFile(self.local, 'r') as zipObj:
                zipObj.extractall(unzipped_folder_dir)
            os.remove(self.local)
        else:
            pass

    def _sha256sum(self):
        """method to check autheticity and completeness of data files;
        using sha256 checksum
        """
        # TODO: Implement method to use checksum
        pass


# %% Example
if __name__ == '__main__':
    # This test requires access to the private topfarm group on Sophia
    # // sophia_dir = '/groups/topfarm/topfarmequinor/dtu10mw_single_rotor/'
    # // filename = 'DTU10MW.nc'
    # This test only requires access to Sophia
    # Setup file source and destination
    home = os.path.expanduser('~')
    filename = 'slurm.conf.example'
    sophia_dir = "/etc/slurm/"
    destination_dir = f'{home}/data/'
    os.makedirs(destination_dir, exist_ok=True)
    local = f"{destination_dir}/{filename}"
    options = {}

    # Use SSH key if you have one, alternatively if no key path is given you will be prompted for a password
    if 0:
        pkey = f"{home}/.ssh/pkey/Sophia"
        options = {'pkey': pkey}

    # Remove test file if present before testing
    if os.path.exists(local):
        os.remove(local)
    # Run command twice to get both possible outputs (1: download file, 2: file already present)
    for i in range(2):
        print(f"\n #{i+1} __init__ of data_manager:")
        dm = data_manager(local=local,
                          source=f"{sophia_dir}/{filename}",
                          platform='Sophia',
                          platform_options=options
                          )
    data_path = dm.data_path
# %%
