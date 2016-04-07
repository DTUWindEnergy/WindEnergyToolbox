
Configure Wine for HAWC2
------------------------

> Note that the steps described here are executed automatically by the
configuration script [```config-wine-hawc2.sh```]
(https://gitlab.windenergy.dtu.dk/toolbox/pbsutils/blob/master/config-wine-hawc2.sh)
in ```pbsutils```.


You will also need to configure wine and place the HAWC2 executables in a
directory that wine knows about. First, activate the correct wine environment by
typing in a shell in the Gorm's home directory (it can be activated with
ssh (Linux, Mac) or putty (MS Windows)):

```
g-000 $ WINEARCH=win32 WINEPREFIX=~/.wine32 wine test.exe
```

Optionally, you can also make an alias (a short format for a longer, more complex
command). In the ```.bashrc``` file in your home directory
(```/home/$USER/.bash_profile```), add at the bottom of the file:

```
alias wine32='WINEARCH=win32 WINEPREFIX=~/.wine32 wine'
```

Add a folder called ```bin``` to your wine system's PATH so we can copy all
the HAWC2 executables in here:

```
WINEDIRNAME=".wine32"
WINEARCH=win32 WINEPREFIX=~/$WINEDIRNAME wine regedit /E tmp.reg "HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment"
sed -i 's/"PATH"="C:\\\\windows\\\\system32;C:\\\\windows"/"PATH"="C:\\\\windows\\\\system32;C:\\\\windows;C:\\\\bin"/g' tmp.reg
WINEARCH=win32 WINEPREFIX=~/$WINEDIRNAME wine regedit ./tmp.reg
rm ./tmp.reg
```

And now copy all the HAWC2 executables, DLL's (including the license manager)
to your wine directory. You can copy all the required executables, dll's and
the license manager are located at ```/home/MET/hawc2exe```. The following
command will do this copying:

```
g-000 $ cp /home/MET/hawc2exe/* /home/$USER/.wine32/drive_c/bin/
```

Notice that the HAWC2 executable names are ```hawc2-latest.exe```,
```hawc2-118.exe```, etc. By default the latest version will be used and the user
does not need to specify this. However, when you need to compare different version
you can easily do so by specifying which case should be run with which
executable. The file ```hawc2-latest.exe``` will always be the latest HAWC2
version at ```/home/MET/hawc2exe/```. When a new HAWC2 is released you can
simply copy all the files from there again to update.

