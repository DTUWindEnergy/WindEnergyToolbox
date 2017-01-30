'''
Created on 23. dec. 2016

@author: mmpe
'''
import unittest
from wetb.utils.cluster_tools.ssh_client import SSHClient
import os

try:
    import sys
    import wetb
    sys.path.append(os.path.dirname(wetb.__file__) + "/../../x")
    import x
except:
    x=None

import io
from wetb.utils.timing import print_time
import shutil
import getpass

class sshrisoe_interactive_auth_handler(object):
    def __init__(self, password):
        self.password = password
              
    def __call__(self, title, instructions, prompt_list):
        if prompt_list:
            if prompt_list[0][0]=="AD Password: ":
                return [self.password]
            return [getpass.getpass(prompt_list[0][0])]
        return []
    
tfp = os.path.join(os.path.dirname(__file__), 'test_files/')
all = 0
class TestSSHClient(unittest.TestCase):

    def setUp(self):
        if x:
            self.ssh = SSHClient('gorm', 'mmpe',x.mmpe )


    def test_execute(self):
        if 0 or all:
            if x:
                _,out,_ = self.ssh.execute("ls -a")
                ssh_ls = ";".join(sorted(out.split("\n"))[3:]) #Exclude ['', '.', '..']
                win_ls = ";".join(sorted(os.listdir(r"z:")))
                self.assertEqual(ssh_ls, win_ls)

    def test_file_transfer(self):
        if 0 or all:
            if x:
                self.ssh.execute("rm -f tmp.txt")
                io.StringIO()
                
                txt = "Hello world"
                f = io.StringIO(txt)
                f.seek(0)
                self.ssh.upload(f, "tmp.txt")
                _,out,_ = self.ssh.execute("cat tmp.txt")
                self.assertEqual(out, txt)
                fn = tfp + "tmp.txt"
                if os.path.isfile (fn):
                    os.remove(fn)
                self.assertFalse(os.path.isfile(fn))
                self.ssh.download("tmp.txt", fn)
                with open(fn) as fid:
                    self.assertEqual(fid.read(), txt)
        
    
    def test_folder_transfer(self):
        if 0 or all:
            if x:
                p = r"C:\mmpe\HAWC2\models\version_12.3beta/"
                p = r'C:\mmpe\programming\python\WindEnergyToolbox\wetb\hawc2\tests\test_files\simulation_setup\DTU10MWRef6.0_IOS/'
                self.ssh.execute("rm -r -f ./tmp_test")
                self.ssh.upload_files(p, "./tmp_test", ["input/"])
                shutil.rmtree("./test/input", ignore_errors=True)
                self.ssh.download_files("./tmp_test", tfp, "input/" )
                os.path.isfile(tfp + "/input/data/DTU_10MW_RWT_Blade_st.dat")
                shutil.rmtree("./test/input", ignore_errors=True)
                 
        
    def test_folder_transfer_specific_files_uppercase(self):
        if 0 or all:
            if x:
                p = tfp
                files = [os.path.join(tfp, "TEST.txt")]
                self.ssh.execute("rm -r -f ./tmp_test")
                self.ssh.upload_files(p, "./tmp_test", file_lst=files)
                self.assertFalse(self.ssh.file_exists("./tmp_test/test.txt"))
                self.assertTrue(self.ssh.file_exists("./tmp_test/TEST.txt"))
                
            
            
    def test_folder_transfer_specific_files(self):
        if 0 or all:
            if x:
                p = r"C:\mmpe\HAWC2\models\version_12.3beta/"
                p = r'C:\mmpe\programming\python\WindEnergyToolbox\wetb\hawc2\tests\test_files\simulation_setup\DTU10MWRef6.0_IOS/'
                files = [os.path.join(os.path.relpath(root, p), f) for root,_,files in os.walk(p+"input/") for f in files]
                self.ssh.execute("rm -r -f ./tmp_test")
                self.ssh.upload_files(p, "./tmp_test", file_lst=files[:5])
                self.ssh.download_files("./tmp_test", tfp + "tmp/", file_lst = files[:3])
                self.assertEqual(len(os.listdir(tfp+"tmp/input/data/")),2)
                shutil.rmtree(tfp + "tmp/")
            
#     def test_ssh_gorm(self):
#         if x:
#             ssh = SSHClient('gorm.risoe.dk', 'mmpe', x.mmpe)
#             _,out,_ = ssh.execute("hostname")
#             self.assertEqual(out.strip(), "g-000.risoe.dk")
            
#     def test_ssh_g047(self):
#         if x:
#             gateway = SSHClient('gorm.risoe.dk', 'mmpe', x.mmpe)
#             ssh = SSHClient('g-047', "mmpe", x.mmpe, gateway=gateway)
#             self.assertEqual(ssh.execute('hostname')[1].strip(), "g-047")

#     def test_ssh_risoe(self):
#         if x:
#             
#             ssh = SSHClient('ssh.risoe.dk', 'mmpe', interactive_auth_handler = sshrisoe_interactive_auth_handler(x.mmpe))
#             _,out,_ = ssh.execute("hostname")
#             self.assertEqual(out.strip(), "ssh-03.risoe.dk")

    def test_ssh_risoe_gorm(self):
        if x:
 
            gateway = SSHClient('ssh.risoe.dk', 'mmpe', interactive_auth_handler = sshrisoe_interactive_auth_handler(x.mmpe))
            ssh = SSHClient('10.40.23.49', 'mmpe', x.mmpe, gateway = gateway)
            _,out,_ = ssh.execute("hostname")
            self.assertEqual(out.strip(), "g-000.risoe.dk")


                
#     def test_ssh_risoe(self):
#         #logger = logging.getLogger("paramiko")
#         #logger.setLevel(logging.DEBUG) # for example
#         #ch = logging.StreamHandler(sys.stdout)
#         #ch.setLevel(logging.DEBUG)
#         #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         #ch.setFormatter(formatter)
#         #logger.addHandler(ch)
#         ssh = SSHClient('ssh.risoe.dk', 'mmpe')
#         print (ssh.connect())
#         return 
#         username = "mmpe"
#  
#         client = paramiko.client.SSHClient()
#          
#         # Any means of getting the PKey will do. This code assumes you've only got one key loaded in your active ssh-agent.
#         # See also:
#         # - http://docs.paramiko.org/en/1.17/api/keys.html#paramiko.pkey.PKey
#         # - http://docs.paramiko.org/en/1.17/api/client.html#paramiko.client.SSHClient.connect
#         my_pkey = None #paramiko.agent.Agent().get_keys()[0]
#          
#         try:
#             client.connect(
#                 hostname="ssh.risoe.dk",
#                 port=22,
#                 username=username,
#                 look_for_keys=False,
#                 pkey=my_pkey
#             )
#         except paramiko.ssh_exception.SSHException as e:
#             pass
#            
#         transport = client.get_transport()
#          
#         # Sometimes sshd is configured to use 'keyboard-interactive' instead of 'password' to implement the YubiKey challenge.
#         # In that case, you can use something like this.
#         # The code below assumes the server will only ask one question and expect the YubiKey OTP as an answer.
#         # If there's more questions to answer, you should handle those per the docs at:
#         #   http://docs.paramiko.org/en/1.17/api/transport.html#paramiko.transport.Transport.auth_interactive
#         # 
#         def interactive_handler(title, instructions, prompt_list):
#             if prompt_list:
#                 if prompt_list[0][0]=="AD Password: ":
#                     return [x.mmpe]
#                 return [getpass.getpass(prompt_list[0][0])]
#             print ("here")
#             return []
#         transport.auth_interactive(username, interactive_handler)
#          
#         #transport.auth_password(username, x.mmpe)
#          
#         # You should now be able to use client as the authenticated user.
#         client.exec_command("echo hej")
#         
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()