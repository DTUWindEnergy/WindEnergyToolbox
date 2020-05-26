'''
Created on 27/11/2015

@author: MMPE
'''

from io import StringIO
import sys
import paramiko

import os
import threading
from _collections import deque
import time
import traceback
import zipfile
import glob
from sshtunnel import SSHTunnelForwarder, SSH_CONFIG_FILE
from wetb.utils.ui import UI
from contextlib import contextmanager
import io
import tempfile


class SSHInteractiveAuthTunnelForwarder(SSHTunnelForwarder):
    def __init__(
            self,
            interactive_auth_handler,
            ssh_address_or_host=None,
            ssh_config_file=SSH_CONFIG_FILE,
            ssh_host_key=None,
            ssh_password=None,
            ssh_pkey=None,
            ssh_private_key_password=None,
            ssh_proxy=None,
            ssh_proxy_enabled=True,
            ssh_username=None,
            local_bind_address=None,
            local_bind_addresses=None,
            logger=None,
            mute_exceptions=False,
            remote_bind_address=None,
            remote_bind_addresses=None,
            set_keepalive=0.0,
            threaded=True,
            compression=None,
            allow_agent=True, *
            args, **
            kwargs):
        self.interactive_auth_handler = interactive_auth_handler
        SSHTunnelForwarder.__init__(self, ssh_address_or_host=ssh_address_or_host, ssh_config_file=ssh_config_file, ssh_host_key=ssh_host_key, ssh_password=ssh_password, ssh_pkey=ssh_pkey, ssh_private_key_password=ssh_private_key_password, ssh_proxy=ssh_proxy, ssh_proxy_enabled=ssh_proxy_enabled, ssh_username=ssh_username,
                                    local_bind_address=local_bind_address, local_bind_addresses=local_bind_addresses, logger=logger, mute_exceptions=mute_exceptions, remote_bind_address=remote_bind_address, remote_bind_addresses=remote_bind_addresses, set_keepalive=set_keepalive, threaded=threaded, compression=compression, allow_agent=allow_agent, *args, **kwargs)

    def _connect_to_gateway(self):
        """
        Open connection to SSH gateway
         - First try with all keys loaded from an SSH agent (if allowed)
         - Then with those passed directly or read from ~/.ssh/config
         - As last resort, try with a provided password
        """
        try:
            self._transport = self._get_transport()
            self._transport.start_client()
            self._transport.auth_interactive(self.ssh_username, self.interactive_auth_handler)
            if self._transport.is_alive:
                return
        except paramiko.AuthenticationException:
            self.logger.debug('Authentication error')
            self._stop_transport()

        self.logger.error('Could not open connection to gateway')

    def _connect_to_gateway_old(self):
        """
        Open connection to SSH gateway
         - First try with all keys loaded from an SSH agent (if allowed)
         - Then with those passed directly or read from ~/.ssh/config
         - As last resort, try with a provided password
        """
        if self.ssh_password:  # avoid conflict using both pass and pkey
            self.logger.debug('Trying to log in with password: {0}'
                              .format('*' * len(self.ssh_password)))
            try:
                self._transport = self._get_transport()
                if self.interactive_auth_handler:
                    self._transport.start_client()
                    self._transport.auth_interactive(self.ssh_username, self.interactive_auth_handler)
                else:
                    self._transport.connect(hostkey=self.ssh_host_key,
                                            username=self.ssh_username,
                                            password=self.ssh_password)

                if self._transport.is_alive:
                    return
            except paramiko.AuthenticationException:
                self.logger.debug('Authentication error')
                self._stop_transport()

        self.logger.error('Could not open connection to gateway')


def unix_path(path):
    drive, p = os.path.splitdrive(str(path))
    return os.path.join(drive, os.path.splitdrive(os.path.realpath(p))[1]).replace("\\", "/")


class SSHClient(object):
    "A wrapper of paramiko.SSHClient"
    TIMEOUT = 4

    def __init__(self, host, username, password=None, port=22, key=None,
                 passphrase=None, interactive_auth_handler=None, gateway=None, ui=UI()):
        self.host = host
        self.username = username
        self.password = password
        if password is None and key is None:
            from os.path import expanduser
            home = expanduser("~")
            key = home + '/.ssh/id_rsa'
        self.port = port
        self.key = key
        self.gateway = gateway
        self.interactive_auth_handler = interactive_auth_handler
        self.ui = ui
        self.disconnect = 0
        self.client = None
        self.ssh_lock = threading.RLock()
        self._sftp = None
        self.transport = None
        self.counter_lock = threading.RLock()
        self.counter = 0
        if key is not None:
            with open(key) as fid:
                self.key = paramiko.RSAKey.from_private_key(fid, password=passphrase)

    def info(self):
        return self.host, self.username, self.password, self.port

    def __enter__(self):
        with self.ssh_lock:
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
        if self.gateway:
            if self.gateway.interactive_auth_handler:
                self.tunnel = SSHInteractiveAuthTunnelForwarder(self.gateway.interactive_auth_handler,
                                                                (self.gateway.host, self.gateway.port),
                                                                ssh_username=self.gateway.username,
                                                                ssh_password=self.gateway.password,
                                                                remote_bind_address=(self.host, self.port),
                                                                local_bind_address=('0.0.0.0', 10022)
                                                                )
            else:
                self.tunnel = SSHTunnelForwarder((self.gateway.host, self.gateway.port),
                                                 ssh_username=self.gateway.username,
                                                 ssh_password=self.gateway.password,
                                                 remote_bind_address=(self.host, self.port),
                                                 local_bind_address=('0.0.0.0', 10022)
                                                 )

            print("start tunnel")
            self.tunnel.start()
            print("self.client = paramiko.SSHClient()")
            self.client = paramiko.SSHClient()

            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            print('self.client.connect("127.0.0.1", 10022, username=self.username, password=self.password)')
            self.client.connect("127.0.0.1", 10022, username=self.username, password=self.password)
            print("done")

        elif self.key is None and (self.password is None or self.password == ""):
            raise IOError("Password not set for %s" % self.host)
        else:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                self.client.connect(self.host, self.port, username=self.username,
                                    password=self.password, pkey=self.key, timeout=self.TIMEOUT)
            except paramiko.ssh_exception.SSHException as e:
                transport = self.client.get_transport()
                transport.auth_interactive(self.username, self.interactive_auth_handler)

        assert self.client is not None
        #self.sftp = paramiko.SFTPClient.from_transport(self.client._transport)
        return self

    def __del__(self):
        self.close()

    @property
    def sftp(self):
        if self._sftp is None:
            self._sftp = paramiko.SFTPClient.from_transport(self.client._transport)
        return self._sftp


#     @sftp.setter
#     def sftp(self, values):
#         pass

    def __exit__(self, *args):
        self.disconnect -= 1
        if self.disconnect == 0:
            self.close()

    def download(self, remotefilepath, localfile, verbose=False, retry=1, callback=None):
        if verbose:
            ret = None
            print("Download %s > %s" % (remotefilepath, str(localfile)))
        if callback is None:
            callback = self.ui.progress_callback()
        remotefilepath = remotefilepath.replace("\\", "/")
        for i in range(retry):
            if i > 0:
                print("Retry download %s, #%d" % (remotefilepath, i))

            try:
                SSHClient.__enter__(self)
                if isinstance(localfile, (str, bytes, int)):
                    ret = self.sftp.get(remotefilepath, localfile, callback=callback)
                elif hasattr(localfile, 'write'):
                    ret = self.sftp.putfo(remotefilepath, localfile, callback=callback)
                break
            except Exception:
                pass
            finally:
                SSHClient.__exit__(self)

            print("Download %s failed from %s" % (remotefilepath, self.host))
        if verbose and ret:
            print(ret)

    def upload(self, localfile, filepath, chmod="770", verbose=False, callback=None):
        if verbose:
            print("Upload %s > %s" % (localfile, filepath))
        if callback is None:
            callback = self.ui.progress_callback()
        try:
            SSHClient.__enter__(self)
            self.execute('mkdir -p "%s"' % (os.path.dirname(filepath)))
            sftp = self.sftp
            if isinstance(localfile, (str, bytes, int)):
                ret = sftp.put(localfile, filepath, callback=callback)
            elif hasattr(localfile, 'read'):
                size = len(localfile.read())
                localfile.seek(0)
                ret = sftp.putfo(localfile, filepath, file_size=size, callback=callback)
            self.execute('chmod %s "%s"' % (chmod, filepath))
        except Exception as e:
            print("upload failed ", str(e))
            raise e
        finally:
            SSHClient.__exit__(self)
        if verbose and ret:
            print(ret)

    def upload_files(self, localpath, remotepath, file_lst=["."], compression_level=1, callback=None):
        assert os.path.isdir(localpath)
        if not isinstance(file_lst, (tuple, list)):
            file_lst = [file_lst]
        files = ([os.path.join(root, f) for fp in file_lst for root, _, files in os.walk(os.path.join(localpath, fp)) for f in files] +
                 [f for fp in file_lst for f in glob.glob(os.path.join(localpath, fp))])
        files = set([os.path.abspath(f) for f in files if os.path.isfile(f)])

        compression_levels = {0: zipfile.ZIP_STORED, 1: zipfile.ZIP_DEFLATED, 2: zipfile.ZIP_BZIP2, 3: zipfile.ZIP_LZMA}
        with self.counter_lock:
            self.counter += 1
            zn = 'tmp_%s_%04d.zip' % (id(self), self.counter)
        zipf = zipfile.ZipFile(zn, 'w', compression_levels[compression_level])
        try:
            for f in files:
                zipf.write(f, os.path.relpath(f, localpath))
            zipf.close()
            remote_zn = os.path.join(remotepath, zn).replace("\\", "/")
            with self:
                self.execute('mkdir -p "%s"' % (remotepath))

                self.upload(zn, remote_zn, callback=callback)
                self.execute('unzip -o "%s" -d "%s" && rm "%s"' % (remote_zn, remotepath, remote_zn))
        except Exception:
            print("upload files failed", )
            traceback.print_exc()
            raise
        finally:
            os.remove(zn)

    def download_files(self, remote_path, localpath, file_lst=["."], compression_level=1, callback=None):
        if not isinstance(file_lst, (tuple, list)):
            file_lst = [file_lst]
        file_lst = [f.replace("\\", "/") for f in file_lst]
        with self.counter_lock:
            self.counter += 1
            zn = 'tmp_%s_%04d.zip' % (id(self), self.counter)

        remote_zip = os.path.join(remote_path, zn).replace("\\", "/")
        self.execute('cd "%s" && zip -r "%s" "%s"' % (remote_path, zn, " ".join(file_lst)))

        local_zip = os.path.join(localpath, zn)
        if not os.path.isdir(localpath):
            os.makedirs(localpath)
        self.download(remote_zip, local_zip, callback=callback)
        self.execute('rm -f "%s"' % remote_zip)
        with zipfile.ZipFile(local_zip, "r") as z:
            z.extractall(localpath)
        os.remove(local_zip)

    def close(self):
        for x in ["_sftp", "client", "tunnel"]:
            try:
                getattr(self, x).close()
                setattr(self, x, None)
            except Exception:
                pass
        self.disconnect = False

    def file_exists(self, filename):
        _, out, _ = (self.execute(
            '[ -f "%s" ] && echo "File exists" || echo "File does not exists"' % unix_path(filename)))
        return out.strip() == "File exists"

    def isfile(self, filename):
        return self.file_exists(filename)

    def folder_exists(self, folder):
        _, out, _ = (self.execute(
            '[ -d "%s" ] && echo "Folder exists" || echo "Folder does not exists"' % unix_path(folder)))
        return out.strip() == "Folder exists"

    def isdir(self, folder):
        return self.folder_exists(folder)

    def execute(self, command, cwd='.', sudo=False, verbose=False):
        feed_password = False
        if sudo and self.username != "root":
            command = "sudo -S -p '' %s" % command
            feed_password = self.password is not None and len(self.password) > 0
        if isinstance(command, (list, tuple)):
            command = "\n".join(command)
        #cwd = unix_path(cwd)
        if verbose:
            print("[%s]$ %s" % (cwd, command))

        command = 'cd "%s" && %s' % (cwd, command)
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
            raise Warning("out:\n%s\n----------\nerr:\n%s" % (out, err))
        elif verbose:
            if out:
                sys.stdout.write(out)
            if err:
                sys.stderr.write(err)
        return v, out, err

    def append_wine_path(self, path):
        ret = self.execute(
            r'wine regedit /E tmp.reg "HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment"')
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

    def glob(self, filepattern, cwd="", recursive=False):
        if isinstance(filepattern, list):
            with self:
                return [f for fp in filepattern for f in self.glob(fp, cwd, recursive)]
        cwd = os.path.join(cwd, os.path.split(filepattern)[0]).replace("\\", "/")
        filepattern = os.path.split(filepattern)[1]
        if recursive:
            _, out, _ = self.execute(r'find "%s" -type f -name "%s"' % (cwd, filepattern))
        else:
            _, out, _ = self.execute(r'find "%s" -maxdepth 1 -type f -name "%s"' % (cwd, filepattern))
        return [file for file in out.strip().split("\n") if file != ""]

    def listdir(self, folder):
        _, out, _ = self.execute('ls -p', cwd=folder)
        return out.split()

    @contextmanager
    def open(self, filename, mode='r', encoding=None, newline=None):
        with tempfile.NamedTemporaryFile(mode='x', delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp_name = tmp.name

        if mode in 'ra+':
            #            try:
            self.download(filename, tmp_name)
        try:
            fid = open(tmp_name, mode=mode, encoding=encoding, newline=newline)
            yield fid

        finally:
            fid.close()
            if mode in 'wa+':
                self.upload(tmp_name, filename)
            os.remove(tmp_name)


class SharedSSHClient(SSHClient):
    def __init__(self, host, username, password=None, port=22, key=None,
                 passphrase=None, interactive_auth_handler=None, gateway=None):
        SSHClient.__init__(self, host, username, password=password, port=port, key=key,
                           passphrase=passphrase, interactive_auth_handler=interactive_auth_handler, gateway=gateway)

        self.shared_ssh_queue = deque()
        self.next = None

    def execute(self, command, sudo=False, verbose=False):
        res = SSHClient.execute(self, command, sudo=sudo, verbose=verbose)
        return res

    def __enter__(self):
        with self.ssh_lock:
            SSHClient.__enter__(self)
            #print ("request SSH", threading.currentThread())
#             if len(self.shared_ssh_queue)>0 and self.shared_ssh_queue[0] == threading.get_ident():
#                 # SSH already allocated to this thread ( multiple use-statements in "with ssh:" block
#                 self.shared_ssh_queue.appendleft(threading.get_ident())
#             else:
#                 self.shared_ssh_queue.append(threading.get_ident())

            if len(self.shared_ssh_queue) > 0 and self.shared_ssh_queue[0] == threading.get_ident():
                # SSH already allocated to this thread ( multiple use-statements in "with ssh:" block
                self.shared_ssh_queue.popleft()

            self.shared_ssh_queue.append(threading.get_ident())

        while self.shared_ssh_queue[0] != threading.get_ident():
            time.sleep(2)

        return self.client

    def __exit__(self, *args):
        with self.ssh_lock:
            if len(self.shared_ssh_queue) > 0 and self.shared_ssh_queue[0] == threading.get_ident():
                self.shared_ssh_queue.popleft()


if __name__ == "__main__":
    try:
        import x
        username, password = 'mmpe', x.password
        client = SSHClient(host='jess.dtu.dk', port=22, username=username, password=password)
        print(client.execute("echo hello $USER from $HOSTNAME")[1])
    except ImportError:
        x = None
