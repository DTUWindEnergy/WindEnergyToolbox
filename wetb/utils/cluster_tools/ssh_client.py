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
import zipfile
from wetb.utils.timing import print_time
import glob
import getpass
from sshtunnel import SSHTunnelForwarder, SSH_CONFIG_FILE



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
        SSHTunnelForwarder.__init__(self, ssh_address_or_host=ssh_address_or_host, ssh_config_file=ssh_config_file, ssh_host_key=ssh_host_key, ssh_password=ssh_password, ssh_pkey=ssh_pkey, ssh_private_key_password=ssh_private_key_password, ssh_proxy=ssh_proxy, ssh_proxy_enabled=ssh_proxy_enabled, ssh_username=ssh_username, local_bind_address=local_bind_address, local_bind_addresses=local_bind_addresses, logger=logger, mute_exceptions=mute_exceptions, remote_bind_address=remote_bind_address, remote_bind_addresses=remote_bind_addresses, set_keepalive=set_keepalive, threaded=threaded, compression=compression, allow_agent=allow_agent, *args, **kwargs)
        

    def _connect_to_gateway(self):
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

class SSHClient(object):
    "A wrapper of paramiko.SSHClient"
    TIMEOUT = 4

    def __init__(self, host, username, password=None, port=22, key=None, passphrase=None, interactive_auth_handler=None, gateway=None):
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.key = key
        self.gateway=gateway
        self.interactive_auth_handler = interactive_auth_handler
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
#        if self.password is None or self.password == "":
#             raise IOError("Password not set for %s"%self.host)
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
            
            self.tunnel.start()
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.client.connect("127.0.0.1", 10022, username=self.username, password=self.password)

                 
        else:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            if self.interactive_auth_handler:
                transport = self.client.get_transport()
                transport.auth_interactive(self.username, self.interactive_handler)
            else:
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
            ret = None
            print ("Download %s > %s" % (remotefilepath, str(localfile)))
        with self:
            for i in range(retry):
                if i>0:
                    print ("Retry download %s, #%d"%(remotefilepath, i))
                try:
                    if isinstance(localfile, (str, bytes, int)):
                        ret = self.sftp.get(remotefilepath, localfile)
                    elif hasattr(localfile, 'write'):
                        ret = self.sftp.putfo(remotefilepath, localfile)
                    break
                except:
                    pass
                print ("Download %s failed from %s"%(remotefilepath, self.host))
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
            
        
    def upload_files(self, localpath, remotepath, file_lst=["."], compression_level=1):
        assert os.path.isdir(localpath)
        if not isinstance(file_lst, (tuple, list)):
            file_lst = [file_lst]
        files = ([os.path.join(root, f) for fp in file_lst for root,_,files in os.walk(os.path.join(localpath, fp )) for f in files] + 
                [f for fp in file_lst for f in glob.glob(os.path.join(localpath, fp)) ])
        files = set([os.path.abspath(f) for f in files])

        compression_levels = {0:zipfile.ZIP_STORED, 1:zipfile.ZIP_DEFLATED, 2:zipfile.ZIP_BZIP2, 3:zipfile.ZIP_LZMA}
        zn =  'tmp_%s_%s.zip'%(id(self),time.time())
        zipf = zipfile.ZipFile(zn, 'w', compression_levels[compression_level])
        try:
            for f in files:
                zipf.write(f, os.path.relpath(f, localpath))
            zipf.close()
            remote_zn = os.path.join(remotepath, zn).replace("\\","/")
            self.execute("mkdir -p %s"%(remotepath))
            
            self.upload(zn, remote_zn)
            self.execute("unzip %s -d %s && rm %s"%(remote_zn, remotepath, remote_zn))
        except:
            raise
        finally:
            os.remove(zn)
        
    
    def download_files(self, remote_path, localpath, file_lst=["."], compression_level=1):
        if not isinstance(file_lst, (tuple, list)):
            file_lst = [file_lst]
        file_lst = [f.replace("\\","/") for f in file_lst]
        remote_zip = os.path.join(remote_path, "tmp.zip").replace("\\","/")
        self.execute("cd %s && zip -r tmp.zip %s"%(remote_path, " ".join(file_lst)))
        
        local_zip = os.path.join(localpath, "tmp.zip")
        if not os.path.isdir(localpath):
            os.makedirs(localpath)
        self.download(remote_zip, local_zip)
        self.execute("rm -f %s" % remote_zip)
        with zipfile.ZipFile(local_zip, "r") as z:
            z.extractall(localpath)
        os.remove(local_zip)
        

    def close(self):
        for x in ["sftp", "client", 'tunnel' ]:
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

    def glob(self, filepattern, cwd="", recursive=False):
        cwd = os.path.join(cwd, os.path.split(filepattern)[0]).replace("\\", "/")
        filepattern = os.path.split(filepattern)[1]
        if recursive:
            _, out, _ = self.execute(r'find %s -type f -name "%s"' % (cwd, filepattern))
        else:
            _, out, _ = self.execute(r'find %s -maxdepth 1 -type f -name "%s"' % (cwd, filepattern))
        return [file for file in out.strip().split("\n") if file != ""]




class SharedSSHClient(SSHClient):
    def __init__(self, host, username, password=None, port=22, key=None, passphrase=None, interactive_auth_handler=None, gateway=None):
        SSHClient.__init__(self, host, username, password=password, port=port, key=key, passphrase=passphrase, interactive_auth_handler=interactive_auth_handler, gateway=gateway)
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
