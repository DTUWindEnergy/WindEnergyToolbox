# Data Manager
The datamanger is intended to alleviate git hosted projects of large data files.
The datamanager curerntly supports two data hosting platforms "Zenodo" and "Sophia".

## Zenodo platform
[Zenodo](https://zenodo.org/) is a general-purpose open repository developed under the European OpenAIRE program and operated by CERN. It allows researchers to deposit research papers, data sets, research software, reports, and any other research related digital artefacts. For each submission, a persistent digital object identifier (DOI) is minted, which makes the stored items easily citeable. ([wiki](https://en.wikipedia.org/wiki/Zenodo)).

The below code snippet shows an example of how to implement the data manager unsing the Zenodo platform in your code.
```
from wetb.utils.cluster_tools.data_manager import data_manager

DOI = '10.5281/zenodo.2562662'       # PyWake project
dm = data_manager(local=f"./data/",
                  source=DOI,
                  platform='Zenodo'
                  )
```
The flowchart below shows the actions of the data manager when instantiated.

<div class="mermaid">
flowchart LR;
    dm(Zenodo DOI \n Local path) --> file_local{File in\n local path?}
    file_local -->|Yes| file_intact

    file_local -->|No| download(Download)
    zenodo[(Zenodo API)] --> download
    download -.-> zenodo

    download-->file_intact{File\n intact?};
    file_intact <-.-> |Checksum| zenodo
    file_intact ---->|Yes| ok[OK]
    file_intact --->|No| re_download(Re-download)
    re_download --> |1x| file_intact
 </div>

## Sophia platform
Sophia is a HPC cluster at DTU. The data manager Sophia platform is intended for internal projects at DTU, the work on the platform implementation was originally intended as an alternative to Git Large File Storage due to its max size of 10GB, but can be used with any data size.

The Sophia data manager implementation uses the WETB's SSH client.

The Sophia platform is a work in progress but works as illustrated by the code snippet and flowchart below:

```
    # Use SSH key if you have one, alternatively if no key path is given you will be prompted for a password
    if 0:
        pkey = f"{home}/.ssh/pkey/Sophia"
        options = {'pkey': pkey}

    dm = data_manager(filename='slurm.conf.example',
                        local='./data/',
                        source='/etc/slurm/',
                        platform='Sophia',
                        platform_options=options
                        )
    data_path = dm.data_path
```

<div class="mermaid">
flowchart LR;
    dm(Sophia path \n Local path) --> run_local{Running\n on Sophia?}
    run_local -->|Yes| ok(OK)
    run_local -->|No| present{File in\n local path?}
    
    present -->|Yes| ok
    present -->|No| download(Download)
    download --> ok


 </div>
