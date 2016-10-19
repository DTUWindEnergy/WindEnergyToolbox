'''
Created on 27/11/2015

@author: MMPE
'''

from io import StringIO
import paramiko
import os
import sys
import threading
from _collections import deque
import time
import traceback

class SSHClient(object):
    "A wrapper of paramiko.SSHClient"
    TIMEOUT = 4

    def __init__(self, host, username, password=None, port=22, key=None, passphrase=None):
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.key = key
        self.disconnect = 0
        self.client = None
        self.sftp = None
        self.transport = None
        if key is not None:
            self.key = paramiko.RSAKey.from_private_key(StringIO(key), password=passphrase)

    def info(self):
        return self.host, self.username, self.password, self.port

    def __enter__(self):
        self.disconnect += 1
        if self.client is None or self.client._transport is None or self.client._transport.is_active() is False:
            self.close()
            try:
                self.connect()
                self.disconnect = 1
            except Exception as e:
                self.close()
                self.disconnect = 0
                raise e
        return self.client

    def connect(self):
        if self.password is None or self.password == "":
            raise IOError("Password not set")
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(self.host, self.port, username=self.username, password=self.password, pkey=self.key, timeout=self.TIMEOUT)
        assert self.client is not None
        self.sftp = paramiko.SFTPClient.from_transport(self.client._transport)
        return self

    def __exit__(self, *args):
        self.disconnect -= 1
        if self.disconnect == 0:
            self.close()


    def download(self, remotefilepath, localfile, verbose=False, retry=1):
        if verbose:
            print ("Download %s > %s" % (remotefilepath, str(localfile)))
        with self:
            for i in range(retry):
                try:
                    if isinstance(localfile, (str, bytes, int)):
                        ret = self.sftp.get(remotefilepath, localfile)
                    elif hasattr(localfile, 'write'):
                        ret = self.sftp.putfo(remotefilepath, localfile)
                    break
                except:
                    pass
                print ("retry", i)
        if verbose:
            print (ret)


    def upload(self, localfile, filepath, verbose=False):
        if verbose:
            print ("Upload %s > %s" % (localfile, filepath))
        with self:
            if isinstance(localfile, (str, bytes, int)):
                ret = self.sftp.put(localfile, filepath)
            elif hasattr(localfile, 'read'):
                ret = self.sftp.putfo(localfile, filepath)
        if verbose:
            print (ret)

    def close(self):
        for x in ["sftp", "client" ]:
            try:
                getattr(self, x).close()
                setattr(self, x, None)
            except:
                pass
        self.disconnect = False

    def file_exists(self, filename):
        _, out, _ = (self.execute('[ -f %s ] && echo "File exists" || echo "File does not exists"' % filename.replace("\\", "/")))
        return out.strip() == "File exists"

    def execute(self, command, sudo=False, verbose=False):
        feed_password = False
        if sudo and self.username != "root":
            command = "sudo -S -p '' %s" % command
            feed_password = self.password is not None and len(self.password) > 0
        if isinstance(command, (list, tuple)):
            command = "\n".join(command)

        if verbose:
            print (">>> " + command)
        with self as ssh:
            if ssh is None:
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                raise Exception("ssh_client exe ssh is None")
            stdin, stdout, stderr = ssh.exec_command(command)
            if feed_password:
                stdin.write(self.password + "\n")
                stdin.flush()

            v, out, err = stdout.channel.recv_exit_status(), stdout.read().decode(), stderr.read().decode()


        if v:
            raise Warning ("out:\n%s\n----------\nerr:\n%s" % (out, err))
        elif verbose:
            if out:
                sys.stdout.write(out)
            if err:
                sys.stderr.write(err)
        return v, out, err

    def append_wine_path(self, path):
        ret = self.execute('wine regedit /E tmp.reg "HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment"')
        self.download('tmp.reg', 'tmp.reg')
        with open('tmp.reg') as fid:
            lines = fid.readlines()

        path_line = [l for l in lines if l.startswith('"PATH"=')][0]
        for p in path_line[8:-1].split(";"):
            if os.path.abspath(p) == os.path.abspath(p):
                return
        if path not in path_line:
            path_line = path_line.strip()[:-1] + ";" + path + '"'

            with open('tmp.reg', 'w') as fid:
                fid.write("".join(lines[:3] + [path_line]))
            self.upload('tmp.reg', 'tmp.reg')
            ret = self.execute('wine regedit tmp.reg')

    def glob(self, filepattern, cwd=""):
        cwd = os.path.join(cwd, os.path.split(filepattern)[0]).replace("\\", "/")
        filepattern = os.path.split(filepattern)[1]
        _, out, _ = self.execute(r'find %s -maxdepth 1 -type f -name "%s"' % (cwd, filepattern))
        return [file for file in out.strip().split("\n") if file != ""]


class SharedSSHClient(SSHClient):
    def __init__(self, host, username, password=None, port=22, key=None, passphrase=None):
        SSHClient.__init__(self, host, username, password=password, port=port, key=key, passphrase=passphrase)
        self.shared_ssh_lock = threading.RLock()
        self.shared_ssh_queue = deque()
        self.next = None


    def execute(self, command, sudo=False, verbose=False):
        res = SSHClient.execute(self, command, sudo=sudo, verbose=verbose)
        return res

    def __enter__(self):
        with self.shared_ssh_lock:
            if self.next == threading.currentThread():
                return self.client
            self.shared_ssh_queue.append(threading.current_thread())
            if self.next is None:
                self.next = self.shared_ssh_queue.popleft()

        while self.next != threading.currentThread():
            time.sleep(1)
        SSHClient.__enter__(self)
        return self.client

    def __exit__(self, *args):
        with self.shared_ssh_lock:
            if next != threading.current_thread():
                with self.shared_ssh_lock:
                    if len(self.shared_ssh_queue) > 0:
                        self.next = self.shared_ssh_queue.popleft()
                    else:
                        self.next = None

if __name__ == "__main__":
    from mmpe.ui.qt_ui import QtInputUI
    q = QtInputUI(None)
    x = None
    username, password = "mmpe", x.password  #q.get_login("mmpe")


    client = SSHClient(host='gorm', port=22, username=username, password=password)
    print (client.glob("*.*", ".hawc2launcher/medium1__1__"))
    #    ssh.upload('../News.txt', 'news.txt')
