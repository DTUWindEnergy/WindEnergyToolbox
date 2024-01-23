#%%

from wetb.utils.envelope import projected_extremes, compute_ensemble_2d_envelope
import numpy as np
import numpy.testing as npt


def test_projected_extremes_basic():
    x = np.cos(np.deg2rad(np.linspace(0,360,12, endpoint=False)))
    y = np.sin(np.deg2rad(np.linspace(0,360,12, endpoint=False)))
    
    res = projected_extremes(
        np.vstack([x,y]).T,
        angles= np.linspace(-150, 180, 12),
        degrees=True
    )
    
    npt.assert_allclose(res[:,1],1.0)
    
    
    
def test_projected_extremes_radians():
    x = np.cos(np.linspace(np.pi/4,9*np.pi/4,4, endpoint=False))
    y = np.sin(np.linspace(np.pi/4,9*np.pi/4,4, endpoint=False))

    res = projected_extremes(
        signal=np.vstack([x,y]).T,
        angles=np.linspace(0,2*np.pi,4,endpoint=False),
        degrees=False
    )
    print(res)

    npt.assert_allclose(res[:,1],np.sqrt(2)/2)


def test_projected_extremes_sectors():
    x = np.cos(np.linspace(0,360,12, endpoint=False))
    y = np.sin(np.linspace(0,360,12, endpoint=False))

    x[x>0] = 0

    res = projected_extremes(
        signal=np.vstack([x,y]).T,
        degrees=True,
        sweep_angle=30
    )

    npt.assert_allclose(res[(res[:,0]> - 75) * (res[:,0] < 75) ][:,1],0.0)



# %%
