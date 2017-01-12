'''
Created on 30/04/2013

@author: Mads M. Pedersen (mmpe@dtu.dk)
'''

from ftplib import FTP
from zipfile import ZipFile
import io
import getpass


class FtpFile():

    def __init__(self, url, username, password):
        if url.lower().startswith('ftp://'):
            url = url[6:]
        self.url = url
        self.domain = url.split('/')[0]
        self.path = '/'.join(url.split('/')[1:])

        self.username = username
        if password is None:
            passwrod = getpass.getpass("Enter password for ftp server")
        self.password = password

    def load(self, open_file=None):
        ftp = FTP(self.domain)
        try:
            ftp.login(self.username, self.password)
            if open_file is None:
                open_file = self
            ftp.retrbinary('RETR /%s' % self.path, open_file.write)
        finally:
            ftp.close()
        open_file.seek(0)

    def write(self, obj):
        raise NotImplementedError

    def __str__(self):
        return "%s(url='%s', username='%s', password=None)" % (self.__class__.__name__, self.url, self.username)

    def name(self):
        return self.path.split('/')[-1]

    def filename(self):
        return self.path


class FtpTxtFile(io.StringIO, FtpFile):

    def __init__(self, url, username, password):
        FtpFile.__init__(self, url, username, password)
        io.StringIO.__init__(self)
        self.load()


    def write(self, s):
        if isinstance(s, bytes):
            s = s.decode('cp1251')
        io.StringIO.write(self, s)



class FtpZipFile(ZipFile, FtpFile):

    def __init__(self, url, username, password):
        FtpFile.__init__(self, url, username, password)
        byteIO = io.BytesIO()
        self.load(byteIO)
        ZipFile.__init__(self, byteIO)

    def FtpZipTxtFile(self, txt_filename):
        return FtpZipTxtFile(self.url, self.username, self.password, txt_filename)


class FtpZipTxtFile(io.StringIO, FtpZipFile):

    def __init__(self, url, username, password, filename):
        FtpZipFile.__init__(self, url, username, password)
        io.StringIO.__init__(self, buf='')
        fid = self.open(filename)
        self.buf = fid.read()
        if self.buf[-1] != "\n":
            self.buf += "\n"
        fid.close()
        self.filename = filename

    def __str__(self):
        return "%s(url='%s', username='%s', password=None, filename='%s')" % (self.__class__.__name__, self.url, self.username, self.filename)


class FtpBinaryFile(io.BytesIO, FtpFile):

    def __init__(self, url, username, password):
        FtpFile.__init__(self, url, username, password)
        self.load()

    def __str__(self):
        return FtpFile.__str__(self)
