from wetb.utils.cluster_tools.pbsfile import PBSFile, Template
import os
from wetb.utils.cluster_tools.os_path import pjoin, relpath, abspath,\
    cluster_path, repl

template = Template("""

[copy_hawc2]

#===============================================================================
echo copy input
#===============================================================================

cd "[modelpath]"
(flock -x 200
[copy_input_to_scratch]
) 200>/scratch/$USER/$PBS_JOBID/[modelname]/lock_file_model
cd "/scratch/$USER/$PBS_JOBID/[modelname]"
[copy_input_to_exe_dir]


#===============================================================================
echo Run HAWC2
#===============================================================================
cd "/scratch/$USER/$PBS_JOBID/[modelname]/run_[jobname]/[rel_exe_dir]"
[hawc2_cmd] [htc_file]

#===============================================================================
echo Copy output
#===============================================================================
cd "/scratch/$USER/$PBS_JOBID/[modelname]/run_[jobname]"
[copy_output]

rm -r "/scratch/$USER/$PBS_JOBID/[modelname]/run_[jobname]"

echo Done
""")


def wine_cmd(platform='win32', hawc2='hawc2mb.exe', cluster='jess'):
    wine_folder = {'jess': {'win32': '.wine32', 'win64': '.wine'}}[cluster][platform]
    wine_prefix = "WINEARCH=%s WINEPREFIX=~/%s " % (platform, wine_folder)
    if cluster == 'jess':
        s = wine_prefix + "winefix\n"
    else:
        s = ""
    return s + wine_prefix + "wine %s" % hawc2


JESS_WINE32_HAWC2MB = wine_cmd()


class HAWC2PBSFile(PBSFile):
    def __init__(self, hawc2_path, hawc2_cmd, htc_file, exe_dir, input_files,
                 output_files, queue='workq', walltime='00:10:00'):
        self.hawc2_path = hawc2_path
        self.hawc2_cmd = hawc2_cmd
        self.htc_file = htc_file
        self.exe_dir = exe_dir
        self.queue = queue
        self.walltime = walltime

        if not os.path.isabs(htc_file):
            htc_file = pjoin(exe_dir, htc_file)
        else:
            htc_file = repl(htc_file)

        if htc_file not in input_files:
            input_files.append(htc_file)
        self.input_files = [abspath((pjoin(exe_dir, f), abspath(f))[os.path.isabs(f)])
                            for f in input_files]

        self.htc_file = relpath(htc_file, exe_dir)

        self.output_files = [abspath((pjoin(exe_dir, f), abspath(f))[os.path.isabs(f)])
                             for f in output_files]

        drive = os.path.splitdrive(exe_dir)[0]
#         print()
#         print("\n".join(self.output_files))
        common_prefix = os.path.commonprefix(self.input_files + self.output_files).rpartition("/")[0]
        p = abspath(pjoin(exe_dir,
                          relpath(common_prefix, exe_dir)))
        self.modelpath = os.path.join(drive, os.path.splitdrive(p)[1])
        self.modelname = os.path.basename(abspath(self.modelpath))
        self.jobname = os.path.splitext(os.path.basename(htc_file))[0]

        PBSFile.__init__(self, self.modelpath, self.jobname, self.commands, queue, walltime=walltime)

    def commands(self):
        rel_exe_dir = relpath(abspath(self.exe_dir), abspath(self.modelpath))
        copy_input_to_scratch, copy_input_to_exe_dir = self.copy_input()
        return template(copy_hawc2=self.copy_hawc2(),
                        exe_dir=cluster_path(self.exe_dir),
                        copy_input_to_scratch=copy_input_to_scratch,
                        copy_input_to_exe_dir=copy_input_to_exe_dir,
                        rel_exe_dir=rel_exe_dir,
                        hawc2_cmd=self.hawc2_cmd,
                        htc_file=self.htc_file,
                        jobname=self.jobname,
                        copy_output=self.copy_output(),
                        modelpath=cluster_path(self.modelpath),
                        modelname=self.modelname)

    def copy_hawc2(self):
        copy_hawc2 = Template('''#===============================================================================
echo copy hawc2 to scratch
#===============================================================================
(flock -x 200
mkdir -p "/scratch/$USER/$PBS_JOBID/hawc2/"
unzip -u -o -q "[hawc2_path]/"*.zip -d "/scratch/$USER/$PBS_JOBID/hawc2/"
find "[hawc2_path]/"* ! -name *.zip -exec cp -u -t "/scratch/$USER/$PBS_JOBID/hawc2/" {} +
) 200>"/scratch/$USER/$PBS_JOBID/lock_file_hawc2"
mkdir -p "/scratch/$USER/$PBS_JOBID/[modelname]/run_[jobname]/[rel_exe_dir]"
cp "/scratch/$USER/$PBS_JOBID/hawc2/"* "/scratch/$USER/$PBS_JOBID/[modelname]/run_[jobname]/[rel_exe_dir]"''')
        if self.hawc2_path is None:
            return ""
        else:
            return copy_hawc2(hawc2_path=os.path.dirname(cluster_path(self.hawc2_path)))

    def copy_input(self):
        rel_input_files = [relpath(f, self.modelpath) for f in self.input_files]

        copy_input = "\n".join(['mkdir -p "[TARGET]/%s" && cp -u -r "%s" "[TARGET]/%s"' % (os.path.dirname(f), f, os.path.dirname(f))
                                for f in rel_input_files])
        return (copy_input.replace("[TARGET]", "/scratch/$USER/$PBS_JOBID/[modelname]"),
                copy_input.replace("[TARGET]", "/scratch/$USER/$PBS_JOBID/[modelname]/run_[jobname]"))

    def copy_output(self):
        rel_output_files = [relpath(f, self.modelpath) for f in self.output_files]
        return "\n".join(['mkdir -p "[modelpath]/%s" && cp -u -r "%s" "[modelpath]/%s"' % (os.path.dirname(f), f, os.path.dirname(f))
                          for f in rel_output_files])
