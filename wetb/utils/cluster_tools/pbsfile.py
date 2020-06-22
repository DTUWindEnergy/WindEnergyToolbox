import os
import inspect
from wetb.utils.cluster_tools.os_path import cluster_path, pjoin, normpath


class Template():

    def __init__(self, template):
        self.template = template

    def __call__(self, **kwargs):
        s = self.template
        found = True
        while found:
            found = False
            for k, v in dict(kwargs).items():
                if "[%s]" % k in s:
                    found = True
                s = s.replace("[%s]" % k, str(v))
        return s


pbs_template = Template('''### Jobid
#PBS -N [jobname]
### Standard Output
#PBS -o [stdout_filename]
[stderr]
#PBS -W umask=0003
### Maximum wallclock time format HOURS:MINUTES:SECONDS
#PBS -l walltime=[walltime]
#PBS -l nodes=[nodes]:ppn=[ppn]
### Queue name
#PBS -q [queue]
cd "[workdir]"
mkdir -p "stdout"
if [ -z "$PBS_JOBID" ]; then echo "Run using qsub"; exit ; fi
pwd
[commands]
exit
''')


class PBSFile():
    _walltime = "00:30:00"

    def __init__(self, workdir, jobname, commands, queue='workq', walltime='00:10:00', nodes=1, ppn=1, merge_std=True):
        """Description

        Parameters
        ----------
        walltime : int, str
            wall time as string ("hh:mm:ss") or second (integer)

        """
        self.workdir = workdir
        self.jobname = jobname
        self.commands = commands
        self.queue = queue
        self.walltime = walltime
        self.nodes = nodes
        self.ppn = ppn
        self.merge_std = merge_std
        self.stdout_filename = normpath(pjoin(workdir, './stdout/%s.out' % jobname))
        self.filename = "pbs_in/%s.in" % self.jobname

    @property
    def walltime(self):
        return self._walltime

    @walltime.setter
    def walltime(self, walltime):
        if isinstance(walltime, (float, int)):
            from math import ceil
            h = walltime // 3600
            m = (walltime % 3600) // 60
            s = ceil(walltime % 60)
            self._walltime = "%02d:%02d:%02d" % (h, m, s)
        else:
            self._walltime = walltime

    def __str__(self):
        if self.merge_std:
            stderr = "### merge stderr into stdout\n#PBS -j oe"
        else:
            stderr = '### Standard Error\n#PBS -e "./err/[jobname].err"'
        if callable(self.commands):
            commands = self.commands()
        else:
            commands = self.commands
        return pbs_template(workdir=cluster_path(self.workdir),
                            stdout_filename=cluster_path(self.stdout_filename),
                            stderr=stderr,
                            jobname=self.jobname,
                            queue=self.queue,
                            walltime=self.walltime,
                            nodes=self.nodes,
                            ppn=self.ppn,
                            commands=commands)

    def save(self, modelpath=None, filename=None):
        modelpath = modelpath or self.workdir
        self.filename = filename or self.filename
        filename = os.path.join(modelpath, self.filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', newline='\n') as fid:
            fid.write(str(self))
        os.chmod(filename, 0o774)


multirunner_template = Template("""echo "[make_dict]
" | python

for node in `cat $PBS_NODEFILE | sort | uniq`
do

     ssh -T $node << EOF &
cd "[workdir]"
python -c "[start_jobs]
"
EOF
done
wait
""")


class PBSMultiRunner(PBSFile):
    def __init__(self, workdir, queue='workq', walltime='01:00:00', nodes=1, ppn=1, merge_std=True, pbsfiles=None,
                 jobname='pbs_multirunner'):
        if pbsfiles:
            def fmt(pbsfile):
                if isinstance(pbsfile, PBSFile):
                    return pbsfile.filename
                return pbsfile
            self.PBS_FILES = "['" + "',\n             '".join([fmt(pbsfile) for pbsfile in pbsfiles]) + "']"
        else:
            self.PBS_FILES = """[os.path.join(root, f) for root, folders, f_lst in os.walk('.') for f in f_lst if f.endswith('.in')]"""

        commands = multirunner_template(make_dict=self.get_src(self.make_dict),
                                        start_jobs=self.get_src(self.start_jobs),
                                        workdir=cluster_path(workdir)).replace("self.ppn", str(ppn))
        PBSFile.__init__(self, workdir, jobname, commands, queue, walltime=walltime,
                         nodes=nodes, ppn=ppn, merge_std=merge_std)
        self.filename = "%s.%s" % (self.jobname, ("lst", "all")[pbsfiles is None])
        self.pbsfiles = pbsfiles

    def make_dict(self):
        import os
        import glob
        import numpy as np
        import re

        # find available nodes
        with open(os.environ['PBS_NODEFILE']) as fid:
            nodes = set([f.strip() for f in fid.readlines() if f.strip() != ''])
        pbs_files = eval(self.PBS_FILES)

        # Make a list of [(pbs_in_filename, stdout_filename, walltime),...]
        pat = re.compile(r'[\s\S]*#\s*PBS\s+-o\s+(.*)[\s\S]*(\d\d:\d\d:\d\d)[\s\S]*')

        def get_info(f):
            try:
                with open(f) as fid:
                    return (f,) + pat.match(fid.read()).groups()
            except Exception:
                return (f, f.replace('.in', '.out'), '00:30:00')
        pbs_info_lst = map(get_info, pbs_files)

        # sort wrt walltime
        pbs_info_lst = sorted(pbs_info_lst, key=lambda fow: tuple(map(int, fow[2].split(':'))))[::-1]
        # make dict {node1: pbs_info_lst1, ...} and save
        d = dict([(f, pbs_info_lst[i::len(nodes)]) for i, f in enumerate(nodes)])
        with open('pbs.dict', 'w') as fid:
            fid.write(str(d))

    def start_jobs(self):
        import os
        import multiprocessing
        import platform
        import time
        with open('pbs.dict') as fid:
            pbs_info_lst = eval(fid.read())[platform.node()]
        arg_lst = ['echo starting %s && mkdir -p "%s" && env PBS_JOBID=$PBS_JOBID "%s" &> "%s" && echo finished %s' %
                   (f, os.path.dirname(o), f, o, f) for f, o, _ in pbs_info_lst]
        print(arg_lst[0])
        print('Starting %d jobs on %s' % (len(arg_lst), platform.node()))
        pool = multiprocessing.Pool(int('$PBS_NUM_PPN'))
        res = pool.map_async(os.system, arg_lst)
        t = time.time()
        for (f, _, _), r in zip(pbs_info_lst, res.get()):
            print('%-50s\t%s' % (f, ('Errorcode %d' % r, 'Done')[r == 0]))
        print('Done %d jobs on %s in %ds' % (len(arg_lst), platform.node(), time.time() - t))

    def get_src(self, func):
        src_lines = inspect.getsource(func).split("\n")[1:]
        indent = len(src_lines[0]) - len(src_lines[0].lstrip())
        src = "\n".join([l[indent:] for l in src_lines])
        if func == self.make_dict:
            src = src.replace("eval(self.PBS_FILES)", self.PBS_FILES)
        return src


if __name__ == '__main__':
    pbsmr = PBSMultiRunner(workdir="W:/simple1", queue='workq', nodes=2, ppn=10, pbsfiles=['/mnt/t1', "/mnt/t2"])
    print(pbsmr.get_src(pbsmr.make_dict))
    # pbsmr.save("c:/tmp/")
