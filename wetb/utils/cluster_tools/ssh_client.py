'''
Created on 27/11/2015

@author: MMPE
'''

from io import StringIO
import paramiko
import os
import sys

class SSHClient(object):
    "A wrapper of paramiko.SSHClient"
    TIMEOUT = 4

    def __init__(self, host, username, password, port=22, key=None, passphrase=None):
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.key = key
        if key is not None:
            self.key = paramiko.RSAKey.from_private_key(StringIO(key), password=passphrase)

    def info(self):
        return self.host, self.username, self.password, self.port

    def __enter__(self):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(self.host, self.port, username=self.username, password=self.password, pkey=self.key, timeout=self.TIMEOUT)
        assert self.client is not None
        self.transport = paramiko.Transport((self.host, self.port))
        self.transport.connect(username=self.username, password=self.password)
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)
        return self

    def __exit__(self, *args):
        self.close()


    def download(self, remotefilepath, localfile, verbose=False):
        if verbose:
            print ("Download %s > %s" % (remotefilepath, str(localfile)))
        with self:
            if isinstance(localfile, (str, bytes, int)):
                ret = self.sftp.get(remotefilepath, localfile)
            elif hasattr(localfile, 'write'):
                ret = self.sftp.putfo(remotefilepath, localfile)
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
        if self.client is not None:
            self.client.close()
            self.client = None
        self.sftp.close()
        self.transport.close()

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
        with self:
            stdin, stdout, stderr = self.client.exec_command(command)
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
        files = []
        for file in out.strip().split("\n"):
            files.append(file.strip())
        return files


if __name__ == "__main__":
    from mmpe.ui.qt_ui import QtInputUI
    q = QtInputUI(None)
    import x
    username, password = "mmpe", x.password  #q.get_login("mmpe")


    client = SSHClient(host='gorm', port=22, username=username, password=password)
    print (client.glob("*.*", ".hawc2launcher/medium1__1__"))
    #    ssh.upload('../News.txt', 'news.txt')
