'''
Created on 13. jan. 2017

@author: mmpe
'''
import unittest

from scipy import signal

import numpy as np
from wetb.signal.filters import first_order
from wetb.signal.fit._fourier_fit import F2x, x2F, rx2F


class Test_first_order_filters(unittest.TestCase):


    def test_low_pass(self):
        a = np.random.randint(0,100,100).astype(float)
        b = first_order.low_pass(a, 1, 1)
        self.assertLess(b.std(), a.std())
        if 0:
            import matplotlib.pyplot as plt
            plt.plot(a)
            plt.plot(b)
            plt.show()
          
          
    def test_low_pass2(self):
        t = np.linspace(0, 1.0, 2001)
        xlow = np.sin(2 * np.pi * 5 * t)
        xhigh = np.sin(2 * np.pi * 250 * t)
        x = xlow + xhigh
                
        b, a = signal.butter(8, 0.125)
        y = signal.filtfilt(b, a, x, padlen=150)
        self.assertAlmostEqual(np.abs(y - xlow).max(),0,4)
        if 0:
            import matplotlib.pyplot as plt
            plt.plot(x)
            plt.plot(y)
            plt.show()
        

#     def test_low_pass3(self):
#         t = np.linspace(0, 1.0, 2001)
# #         xlow = np.sin(2 * np.pi * 5 * t)
# #         xhigh = np.sin(2 * np.pi * 250 * t)
# #         x = xlow + xhigh
#         x = np.sum([np.sin(t*x*2*np.pi) for x in range(1,200)],0)                
#         cutoff = .2
#         b, a = signal.butter(8,cutoff)
#         w, h = signal.freqs(b, a)
#         y = signal.filtfilt(b, a, x)
#         F = rx2F(x, max_nfft=len(t))
#         tF = np.linspace(0, len(F)/t[-1],len(F))
#         Fy = rx2F(y, max_nfft=len(t))
# 
#         if 1:
#             import matplotlib.pyplot as plt
# #             plt.plot(x)
# #             plt.plot(y)
# #             plt.show()
# 
#             plt.plot(tF, np.abs(F))
#             plt.plot(tF, np.abs(Fy))
#             plt.xlim([0,260])
#             plt.show()
#             print (b,a)
#             
#             plt.plot(w, 20 * np.log10(abs(h)))
#             plt.xscale('log')
#             plt.title('Butterworth filter frequency response')
#             plt.xlabel('Frequency [radians / second]')
#             plt.ylabel('Amplitude [dB]')
#             plt.margins(0, 0.1)
#             plt.grid(which='both', axis='both')
#             plt.axvline(cutoff, color='green') # cutoff frequency
#             plt.show()
#      
#     def test_low_pass2(self):
#         t = np.arange(100000)/10000
#         dt = t[1]-t[0]
#         hz2 = np.sin(t*2*2*np.pi) # frq = 2 hz or 2*2pi rad/s
#         hz10 = np.sin(t*10*2*np.pi) # frq = 10 hz or 10*2pi rad/s
#         hz20 = np.sin(t*100*2*np.pi) # frq = 100 hz or 100*2pi rad/s
#         #sig = np.sum([np.sin(t*x*2*np.pi) for x in range(1,200)],0)
#         sig = np.sum([np.sin(t*x*2*np.pi) for x in [1,5,10]],0)
#         print (sig.shape)
#         F = rx2F(sig, max_nfft=len(t))
#         tF = np.linspace(0, len(F)/t[-1],len(F))
#           
#           
#         sig_lp1 = first_order.low_pass(sig, dt, 30*dt)
#         F_lp1 = rx2F(sig_lp1, max_nfft=len(t))
#    
#         bb, ba = signal.butter(1,5, 'low', analog=True)
#         print (bb, ba)
#         w1, h1 = signal.freqs(bb, ba)
#         sig_lp2 = signal.filtfilt(bb,ba, sig)
# #         print (sig_lp2)
# #         F_lp2 = rx2F(sig_lp2, max_nfft=len(t))
# #         bb, ba = signal.butter(10,10, 'low', analog=True)
# #         sig_lp3 = signal.lfilter(bb,ba, sig)
# #         F_lp3 = rx2F(sig_lp3, max_nfft=len(t))
# #         w2, h2 = signal.freqs(bb, ba)
# # #          
# #         self.assertLess(b.std(), a.std())
#         if 1:
#             import matplotlib.pyplot as plt
#             plt.plot(t, sig)
# #             plt.plot(t, np.sum([1/x*np.sin(t*x*2*np.pi) for x in [1]],0))
# #             
#             plt.plot(t, sig_lp1)
#             plt.plot(t, sig_lp2)
#             #plt.plot(t, sig_lp3)
#               
#             plt.show()
#                
# #             plt.plot(tF, np.abs(F))
# #             plt.plot(tF, np.abs(F_lp1))
# # #             plt.plot(tF, np.abs(F_lp3))
# # #             plt.plot(tF, np.abs(F_lp2))
# #             plt.plot([0,1000],[0.708,.708])              
# #             plt.xlim([0,205])
# #             plt.ylim([0,1])
# #             plt.show()
# #             plt.plot(w1, 20 * np.log10(abs(h1)))
# #             plt.plot(w2, 20 * np.log10(abs(h2)))
# #             plt.xscale('log')
# #             plt.title('Butterworth filter frequency response')
# #             plt.xlabel('Frequency [radians / second]')
# #             plt.ylabel('Amplitude [dB]')
# #             plt.margins(0, 0.1)
# #             plt.grid(which='both', axis='both')
# #             plt.axvline(100, color='green') # cutoff frequency
# #             plt.show()
 
          
 
# #     def test_low_pass2(self):
# #         F = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,.1]
# #         F +=[0]*(len(F)-1)
# #         x = F2x(F)
# #         a = np.tile(x, 3)
# #         print (x.shape)
# # #         a = np.random.randint(0,100,100).astype(float)
# #         b = first_order.low_pass(a, .1, 1)
# #         bb, ba = signal.butter(10,100, 'low', analog=True)
# #         #c = signal.lfilter(bb,ba, a)
# #         w, h = signal.freqs(bb, ba)
# #          
# # #         self.assertLess(b.std(), a.std())
# #         if 1:
# #             import matplotlib.pyplot as plt
# #             plt.plot(a)
# #             #plt.ylim([-2,2])
# #             plt.plot(b)
# #             #plt.plot(c)
# #             plt.show()
# #             print (h)
# #             plt.plot(w, abs(h))
# #             plt.xscale('log')
# #             plt.title('Butterworth filter frequency response')
# #             plt.xlabel('Frequency [radians / second]')
# #             plt.ylabel('Amplitude [dB]')
# #             plt.margins(0, 0.1)
# #             plt.grid(which='both', axis='both')
# #             plt.axvline(100, color='green') # cutoff frequency
# #             plt.show()
# #              
# #             
    

    def test_high_pass(self):
        a = np.random.randint(0,100,100).astype(float)
        b = first_order.high_pass(a, 1, 1)
        self.assertLess(b.mean(), a.mean())
        if 0:
            import matplotlib.pyplot as plt
            plt.plot(a)
            plt.plot(b)
            plt.show()
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()