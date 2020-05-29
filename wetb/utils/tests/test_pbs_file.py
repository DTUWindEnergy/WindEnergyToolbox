from wetb.utils.cluster_tools.pbsfile import PBSFile, Template, PBSMultiRunner
from wetb.utils.cluster_tools import ssh_client
from wetb.utils.cluster_tools.ssh_client import SSHClient
import io
from wetb.utils.cluster_tools.pbsjob import SSHPBSJob, DONE
import time
import pytest
try:
    import x
except ImportError:
    x = None


def test_template():
    t = Template('[a]B[c]')
    assert t(a="A", c="C") == "ABC"
    assert t(a="[c]", c="C") == "CBC", "%s!=%s" % (t(a="[c]", c="C"), 'CBC')


def test_pbs_file_str():
    pbsfile = PBSFile('/home/user/tmp', "test", '''python -c "print('hello world')"''', 'workq')
    ref = """### Jobid
#PBS -N test
### Standard Output
#PBS -o /home/user/tmp/stdout/test.out
### merge stderr into stdout
#PBS -j oe
#PBS -W umask=0003
### Maximum wallclock time format HOURS:MINUTES:SECONDS
#PBS -l walltime=00:10:00
#PBS -l nodes=1:ppn=1
### Queue name
#PBS -q workq
cd "/home/user/tmp"
mkdir -p "stdout"
if [ -z "$PBS_JOBID" ]; then echo "Run using qsub"; exit ; fi
pwd
python -c "print('hello world')"
exit
"""
    assert str(pbsfile) == ref


def test_pbs_file():
    if x is None:
        pytest.xfail("Password missing")
    pbsfile = PBSFile("/home/mmpe/tmp", "test", '''python -c "print('hello world')"''', 'workq')
    ssh = SSHClient("jess.dtu.dk", 'mmpe', x.mmpe)
    pbs_job = SSHPBSJob(ssh)
    pbs_job.submit(pbsfile, "./tmp")
    with pbs_job.ssh:
        start = time.time()
        while time.time() < start + 10:
            time.sleep(.1)
            if pbs_job.status == DONE:
                break
        else:
            raise Exception("job not finished within 10 s")
        _, out, _ = ssh.execute('cat ./tmp/stdout/test.out')
    assert "hello world" in out


@pytest.mark.parametrize('i,s', [("01:02:03", "01:02:03"),
                                 (5, "00:00:05"),
                                 (4000, '01:06:40')])
def test_pbs_walltime(i, s):
    pbsfile = PBSFile("./tmp", "test", '', 'workq', walltime=i)
    assert pbsfile.walltime == s


def test_pbs_multirunner():
    pbs = PBSMultiRunner("/home/user/tmp", )
    ref = r"""### Jobid
#PBS -N pbs_multirunner
### Standard Output
#PBS -o /home/user/tmp/stdout/pbs_multirunner.out
### merge stderr into stdout
#PBS -j oe
#PBS -W umask=0003
### Maximum wallclock time format HOURS:MINUTES:SECONDS
#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=1
### Queue name
#PBS -q workq
cd "/home/user/tmp"
mkdir -p "stdout"
if [ -z "$PBS_JOBID" ]; then echo "Run using qsub"; exit ; fi
pwd
echo "import os
import glob
import numpy as np
import re

# find available nodes
with open(os.environ['PBS_NODEFILE']) as fid:
    nodes = set([f.strip() for f in fid.readlines() if f.strip() != ''])
pbs_files = [os.path.join(root, f) for root, folders, f_lst in os.walk('.') for f in f_lst if f.endswith('.in')]

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

" | python

for node in `cat $PBS_NODEFILE | sort | uniq`
do

     ssh -T $node << EOF &
cd "/home/user/tmp"
python -c "import os
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

"
EOF
done
wait

exit
"""
    assert str(pbs) == ref
