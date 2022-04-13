# %%
from abc import ABC
import os
from wetb.utils.cluster_tools.ssh_client import SSHClient
import getpass
from tqdm import tqdm
import socket
import requests
import urllib.request
import hashlib


class DownloadProgressBarSophia(tqdm):
    """Progressbar for download of data,
    for use with callbacks.
    """
    def update_to(self, current, total):
        self.total = total
        self.update(current - self.n)


class DownloadProgressBarZenodo(tqdm):
    """Progressbar for download of data,
    for use with callbacks.
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class data_manager(ABC):
    """A Data manager to handle large files in shared projects.
    When the data manager is intialized it will check if the data file is present locally.
    If it is not it will be downloaded from a given online source.
    Currently only Sophia is implemented using the "wetb" SSHClient.

    With relatively little effort a different platform can easily be implemented.
    Such as Zenodo or DTU Data.
    """

    def __init__(self, local, source, platform, filename=None, unzip=True, checksum=None, platform_options=None):
        """Initialize data manager, with source and destination

        Parameters
        ----------
        filename : str
            Name of data file
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
        self.name = filename
        self.local = local
        self.source = source
        self.platform = platform.lower()
        self.platform_options = platform_options
        self.checksum = checksum

        # TODO: Re-structure code with specific Sophia __call__ to remove reference to Sophia in __init__
        # Check if already on Sophia
        if 'hpc' in socket.gethostname() and self.platform == 'sophia':
            data_path = f"{self.source}{self.name}"
            print(f"Running on Sophia no download required, data_path set to: {data_path}")
        else:
            # Call main function
            data_path = self.__call__()
            if unzip:
                self._unzip()

        self.data_path = data_path

    def __call__(self):
        """Evaluate if data is prensent locally.
        or alternatively download it.
        """
        if self.platform == 'sophia':
            if os.path.exists(f"{self.local}/{self.name}"):
                print(f"Data present in local directory: {self.local}/{self.name}")
            else:
                self._download_sophia()
                print(f"Data downloaded to local directory: {self.local}/{self.name}")

        elif self.platform == 'zenodo':
            self._get_metadata_zenodo()
            all_checksums_passed = self._md5_checksum()
            if all_checksums_passed:
                print(f"All data in zenodo record already downloaded and validated.")
            else:
                self._download_zenodo()
                print(f"Data downloaded to local directory: {self.local}")

        else:
            raise NotImplementedError(f"{self.platform} Is not a valid platform, \
                                       currently the following platforms are available: 'Sophia' and 'Zenodo'")

        data_path = f"{self.local}/{self.name}"
        return data_path

    def _download_sophia(self):
        """Download data file(s) from the DTU Sophia cluster
        """
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
        with DownloadProgressBarSophia(unit='B', unit_scale=True) as t:
            client.download(f"{self.source}/{self.name}", f"{self.local}/{self.name}", verbose=False, callback=t.update_to)

        # Check file is not empty, as SSH CLient returns an empty file with correct name when failing.
        if os.stat(f"{self.local}/{self.name}").st_size == 0:
            raise RuntimeError("Dowloaded file is empty, please check source is an existsing file.")

    def _get_metadata_zenodo(self):
        """Obtain metadata from zenodo record id
        """
        if not self.platform == 'zenodo':
            raise ValueError("get_metadata_zenodo can only be called with platform = 'Zenodo'")
        api_url = 'https://zenodo.org/api/records/'
        if 'zenodo' in self.source:
            recordID = self.source.split('.')[-1]
        else:
            recordID = self.source
        r = requests.get(api_url + recordID)
        metadata = r.json()
        self.metadata = metadata

    def _download_zenodo(self):
        """Download data file(s) from zenodo.org
        """
        if not hasattr(self, 'metadata'):
            self._get_metadata_zenodo()

        # Function to download the files inside zenodo record id using urllib
        def download_url(url, output_path):
            """Local downloader based on urllib request

            Parameters
            ----------
            url : str
                url to zenodo file
            output_path : str
                output directory including filename
            """
            with DownloadProgressBarZenodo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

        # Run download function on all files in record id
        files = self.metadata['files']
        for i, file in enumerate(files):
            if hasattr(file, 'checksum_correct') and file['checksum_correct']:
                print(f"Skipping file {i+1} of {len(files)}, file downloaded and validated")
            else:
                print(f"Downloading file {i+1} of {len(files)}")
                zenodo_url = file['links']['self']
                os.makedirs('/'.join(f"{self.local}{file['key']}".split('/')[:-1]), exist_ok=True)
                download_url(zenodo_url, f"{self.local}{file['key']}")
                if self._md5(f"{self.local}{file['key']}") == file['checksum'].split(':')[-1]:
                    file['checksum_correct'] = True
                else:
                    os.remove(f"{self.local}{file['key']}")
                    raise RuntimeError(f'Failed validation of file {i+1}, file removed')

    def _unzip(self):
        """Unzip file if zipped.
        # TODO: Remake function to unpack multiple file formats e.g. .tar or .7zip
        """
        def _unzip_local(zipped_dir):
            from zipfile import ZipFile
            # Extract zipfile
            unzipped_folder_dir = zipped_dir.split('.zip')[0]
            os.makedirs(unzipped_folder_dir, exist_ok=True)
            with ZipFile(zipped_dir, 'r') as zipObj:
                zipObj.extractall(unzipped_folder_dir)
            # TODO: Make removal of .zip file possible by storing cheksum in a different way so online resource checksum mathces local
            # //os.remove(zipped_dir)

        if self.platform == 'sophia':
            if self.local.split('.')[-1] == 'zip':
                _unzip_local(self.local)

        elif self.platform == "zenodo":
            for file in self.metadata['files']:
                file_path = f"{self.local}{file['key']}"
                if file_path.split('.')[-1] == 'zip':
                    _unzip_local(file_path)
        else:
            pass

    def _md5(self, file, buf_size=65536):
        """Function to hash a file using md5

        Parameters
        ----------
        file : str
            path to file
        buf_size : int, optional
            size of buffer in memory, by default 65536

        Returns
        -------
        hash : str
            hash value of given file
        """
        hash_md5 = hashlib.md5()
        with open(file, "rb") as f:
            for chunk in iter(lambda: f.read(buf_size), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _md5_checksum(self):
        """method to check autheticity and completeness of data files;
        using md5 checksum
        """

        all_checksums_passed = None

        if self.platform == 'zenodo':
            if not hasattr(self, 'metadata'):
                self._get_metadata_zenodo()
            for file in self.metadata['files']:
                if os.path.exists(f"{self.local}{file['key']}"):
                    all_checksums_passed = True
                    hash_local = self._md5(f"{self.local}{file['key']}")
                    hash_zenodo = file['checksum'].split(':')[-1]
                    file['checksum_correct'] = hash_local == hash_zenodo
                    if not hash_local == hash_zenodo:
                        all_checksums_passed = False
        else:
            pass

        return all_checksums_passed


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
    destination_dir = f'./data/'
    os.makedirs(destination_dir, exist_ok=True)
    options = {}

    # Use SSH key if you have one, alternatively if no key path is given you will be prompted for a password
    if 0:
        pkey = f"{home}/.ssh/pkey/Sophia"
        options = {'pkey': pkey}

    # Remove test file if present before testing
    if os.path.exists(f"{destination_dir}/{filename}"):
        os.remove(f"{destination_dir}/{filename}")
    # Run command twice to get both possible outputs (1: download file, 2: file already present)
    for i in range(2):
        print(f"\n #{i+1} __init__ of data_manager with Sophia: \n")
        dm = data_manager(filename=filename,
                          local=destination_dir,
                          source=sophia_dir,
                          platform='Sophia',
                          platform_options=options
                          )
    data_path = dm.data_path

    # Zenodo example
    # Run command twice to get both possible outputs (1: download file, 2: file already present)
    # DOI = '10.5281/zenodo.3737683'       # PyWake presentation
    DOI = '10.5281/zenodo.2562662'       # PyWake project
    local = f"./data/"
    for i in range(2):
        print(f"\n #{i+1} __init__ of data_manager with Zenodo: \n")
        dm = data_manager(local=local,
                          source=DOI,
                          platform='Zenodo'
                          )
# %%
