'''
Created on 24/04/2014

@author: MMPE
'''
from wetb.hawc2.htc_file import HTCFile
import numpy as np
import os


class ShearFile(object):
    """HAWC2 user defined shear file
    
    Examples:
    ---------
    
    >>> sf = ShearFile([-55, 55], [30, 100, 160] , u=np.array([[0.7, 1, 1.3], [0.7, 1, 1.3]]).T)
    >>> print (sf.uvw([-55,0],[65,135])) #uvw factors
    [array([ 0.85 ,  1.175]), array([ 0.,  0.]), array([ 0.,  0.])]
    >>> from wetb.wind.shear import power_shear
    >>> print (sf.uvw([-55,0],[65,135], shear=power_shear(.2,100,10))) # uvw wind speeds
    [array([  7.79832978,  12.47684042]), array([ 0.,  0.]), array([ 0.,  0.])]
    >>> sf.save('test.dat')
    """
    
    def __init__(self,v_positions, w_positions, u=None, v=None, w=None, u0=None, shear=None):
        """
        Parameters
        ----------
        v_positions : array_like
            lateral coordinates
        w_positions : array_like
            vertical coordinates
        u : array_like, optional
            shear_u component, normalized with U_mean\n
            shape must be (#w_positions, #v_positions) or (#w_positions,)
        v : array_like, optional
            shear_v component, normalized with U_mean\n
            shape must be (#w_positions, #v_positions) or (#w_positions,)
        w : array_like, optional
            shear_w component, normalized with U_mean\n
            shape must be (#w_positions, #v_positions) or (#w_positions,)
        u0 : int or float
            Mean wind speed
        shear : function
            Shear function: f(height)->wsp
        """
        self.v_positions = v_positions
        self.w_positions = w_positions
        shape = (len(w_positions), len(v_positions))
        uvw = [u, v, w]
        for i in range(3):
            if uvw[i] is None:
                if i == 0:
                    uvw[i] = np.ones((shape))
                else:
                    uvw[i] = np.zeros((shape))
            else:
                uvw[i] = np.array(uvw[i])
                if len(uvw[i].shape) == 1 and uvw[i].shape[0] == shape[0]:
                    uvw[i] = np.repeat(np.atleast_2d(uvw[i]).T, shape[1], 1)
    
                assert uvw[i].shape == shape, (i, uvw[i].shape, shape)
        self.u, self.v, self.w = uvw
        self.u0 = u0
        self.shear = shear
    
    def uvw(self, v,w,u0=None,shear=None):
        """Calculate u,v,w wind speeds at position(s) (v,w)
        Parameters
        ----------
        v : int, float or array_like
            v-coordinate(s)
        w : int, float or array_like
            w-coordinates(s)
        u0 : int or float
            mean wind speed
        shear : function or None
            if function: f(height)->wsp
            if None: self.shear is used, if not None. 
            Otherwise wind speed factors instead of absolute wind speeds are returned
            
        Returns
        -------
        u,v,w 
            wind speed(s) or wind speed factor(s) if shear not defined
        """
        u0 =u0 or self.u0 or 1
        shear = shear or self.shear or (lambda z: 0)
        from scipy.interpolate import RegularGridInterpolator
        wv = np.array([w,v]).T
        return [(RegularGridInterpolator((self.w_positions, self.v_positions), uvw)(wv))*u0+shear(w) 
                            for uvw in [self.u, self.v, self.w] if uvw is not None]
        


    def save(self, filename):
        """Save user defined shear file
        Parameters
        ----------
        filename : str:
            Filename
        """
        
        # exist_ok does not exist in Python27
        filename = os.path.abspath(filename)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))#, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as fid:
            fid.write(" # autogenerated shear file\n")
            fid.write("  %d %d\n" % (len(self.v_positions), len(self.w_positions)))
            
            for i, (l,vuw) in enumerate(zip(['v', 'u', 'w'],[self.v,self.u,self.w])):
                fid.write(" # shear %s component\n  " % l)
                fid.write("\n  ".join([" ".join(["%.10f" % v for v in r ]) for r in vuw]))
                fid.write("\n")
            for yz, coor in (['v', self.v_positions], ['w', self.w_positions]):
                fid.write(" # %s coordinates\n  " % yz)
                fid.write("\n  ".join("%.10f" % v for v in coor))
                fid.write("\n")

    @staticmethod
    def load(filename):
        """Load shear file
        Parameters 
        ----------
        filename : str
            Filename
            
        Returns
        -------
        shear file : ShearFile-object
        """
        with open(filename) as fid:
            lines = fid.readlines()
        no_V,no_W=map(int,lines[1].split())
        v,u,w = [np.array([row.split() for row in lines[3+(no_W+1)*i:3+(no_W+1)*(i+1)-1] ],dtype=float) for i in range(3)]
        v_positions = np.array(lines[3+(no_W+1)*3:3+(no_W+1)*3+no_V],dtype=float)
        w_positions = np.array(lines[3+(no_W+1)*3+(no_V+1):3+(no_W+1)*3+(no_V+1)+no_W],dtype=float)
        return ShearFile(v_positions,w_positions,u,v,w)

    @staticmethod
    def load_from_htc(htc_file):
        """Load shear file from HTC file including shear function
        Parameters 
        ----------
        htc_file : str or HTCFile
            Filename or HTCFile
            
        Returns
        -------
        shear file : ShearFile-object
        """
        if isinstance(htc_file,str):
            htc_file = HTCFile(htc_file)
        user_defined_shear_filename = os.path.join(htc_file.modelpath, htc_file.wind.user_defined_shear[0])
        shear_file = ShearFile.load(user_defined_shear_filename)
        shear_file.shear = htc_file.get_shear()
        shear_file.u0 = htc_file.wind.wsp[0]
        return shear_file

def save(filename, v_coordinates, w_coordinates, u=None, v=None, w=None):
    """Save shear file (deprecated)"""
    ShearFile(v_coordinates, w_coordinates,u,v,w).save(filename)





if __name__ == "__main__":
    from wetb.wind.shear import power_shear
    sf = ShearFile([-55, 55], [30, 100, 160] , u=np.array([[0.7, 1, 1.3], [0.7, 1, 1.3]]).T)
    print (sf.uvw([-55,0],[65,135])) #uvw factors
    #[array([ 0.85 ,  1.175]), array([ 0.,  0.]), array([ 0.,  0.])]
    print (sf.uvw([-55,0],[65,135], shear=power_shear(.2,100,10))) # uvw wind speeds
    #[array([  7.79832978,  12.47684042]), array([ 0.,  0.]), array([ 0.,  0.])]
    sf.save('test.dat')
    
