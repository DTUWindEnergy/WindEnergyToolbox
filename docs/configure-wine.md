Background Information Regarding Wine
-------------------------------------

> Note that the steps described here are executed automatically by the
configuration script [```config-wine-hawc2.sh```]
(https://gitlab.windenergy.dtu.dk/toolbox/pbsutils/blob/master/config-wine-hawc2.sh)
in ```pbsutils```.


Configure Wine for Gorm
------------------------

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

Add a folder called ```~/wine_exe/win32``` to your wine system's PATH so we can
copy all the HAWC2 executables in here:

```
$EXE_DIR_WINE="z:/home/$USER/wine_exe/win32/"
printf 'REGEDIT4\n[HKEY_CURRENT_USER\\Environment]\n"PATH"="'"$EXE_DIR_WINE"'"\n' >> ./tmp.reg
WINEARCH=win32 WINEPREFIX=~/.wine32 wine regedit ./tmp.reg
rm ./tmp.reg
```

And now copy all the HAWC2 executables, DLL's (including the license manager)
to your wine directory. You can copy all the required executables, dll's and
the license manager are located at ```/home/MET/hawc2exe```. The following
command will update your local directory with any new executables that have
been placed in ```/home/MET/hawc2exe/win32/```:

```
g-000 $ rsync -a /home/MET/hawc2exe/win32/* /home/$USER/wine_exe/win32/
```

Notice that the HAWC2 executable names are ```hawc2-latest.exe```,
```hawc2-118.exe```, etc. By default the latest version will be used and the user
does not need to specify this. However, when you need to compare different version
you can easily do so by specifying which case should be run with which
executable. The file ```hawc2-latest.exe``` will always be the latest HAWC2
version at ```/home/MET/hawc2exe/```. When a new HAWC2 is released you can
simply copy all the files from there again to update.


Configure Wine for Jess
------------------------

Same principles apply to Jess, and  [```config-wine-hawc2.sh```]
(https://gitlab.windenergy.dtu.dk/toolbox/pbsutils/blob/master/config-wine-hawc2.sh)
can be used to initialize and configure your wine environment.

Note that due to a bug in the specific version of wine that is installed on
Jess, ```config-wine-hawc2.sh``` will apply the following command to fix this.
It is important to note that this fix will have to be executed on each node at
the beginning of each new session:

```
j-000 $ WINEARCH=win32 WINEPREFIX=~/.wine32 winefix
```

```winefix``` is automatically included in the ```pbs_in``` scripts genetrated
by the toolbox.

