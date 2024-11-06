import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def DLB_extreme_loads_to_excel(DLB_extreme_loads, path):
    """
    Generate excel file with the DLB extreme loads.

    Parameters
    ----------
    DLB_extreme_loads : xarray.DataArray ('sensor_name', 'angle', '(value, dlc, group)')
        DataArray containing the extreme loads of the DLB. It matches the
        output from get_DLB_extreme_loads
    path : str
        Path to the new excel file to be created

    Returns
    -------
    None.

    """
    df_DLB_extreme_loads = DLB_extreme_loads.to_dataframe('value').reset_index()
    sensor_name = [df_DLB_extreme_loads['sensor_name'].iloc[i]
                   for i in range(0, len(df_DLB_extreme_loads), 3)]
    sensor_unit = [df_DLB_extreme_loads['sensor_unit'].iloc[i]
                   for i in range(0, len(df_DLB_extreme_loads), 3)]
    angle = [df_DLB_extreme_loads['angle'].iloc[i]
                   for i in range(0, len(df_DLB_extreme_loads), 3)]
    value = [int(float(df_DLB_extreme_loads['value'].iloc[i]))
                   for i in range(0, len(df_DLB_extreme_loads), 3)]
    dlc = [df_DLB_extreme_loads['value'].iloc[i]
                   for i in range(1, len(df_DLB_extreme_loads), 3)]
    group = [df_DLB_extreme_loads['value'].iloc[i]
                   for i in range(2, len(df_DLB_extreme_loads), 3)]
    
    df_DLB_extreme_loads = pd.DataFrame()
    df_DLB_extreme_loads['sensor_name'] = sensor_name
    df_DLB_extreme_loads['sensor_unit'] = sensor_unit
    df_DLB_extreme_loads['angle'] = angle
    df_DLB_extreme_loads['value'] = value
    df_DLB_extreme_loads['dlc'] = dlc
    df_DLB_extreme_loads['group'] = group
    
    df_DLB_extreme_loads.to_excel(path, index=False)

def DLB_fatigue_loads_to_excel(DLB_fatigue_loads, path):
    """
    Generate excel file with the DLB fatigue loads.

    Parameters
    ----------
    DLB_fatigue_loads : xarray.DataArray ('sensor_name', 'angle', 'm')
        DataArray containing the fatigue loads of the DLB. It matches the
        output from get_DLB_fatigue_loads
    path : str
        Path to the new excel file to be created

    Returns
    -------
    None.

    """
    df_DLB_fatigue_loads = DLB_fatigue_loads.to_dataframe('value').reset_index()
    sensor_name = df_DLB_fatigue_loads['sensor_name']
    sensor_unit = df_DLB_fatigue_loads['sensor_unit']
    angle = df_DLB_fatigue_loads['angle']
    m = df_DLB_fatigue_loads['m']
    value = [int(df_DLB_fatigue_loads['value'].iloc[i]) 
             for i in range(len(df_DLB_fatigue_loads))]
    
    df_DLB_fatigue_loads = pd.DataFrame()
    df_DLB_fatigue_loads['sensor_name'] = sensor_name
    df_DLB_fatigue_loads['sensor_unit'] = sensor_unit
    df_DLB_fatigue_loads['angle'] = angle
    df_DLB_fatigue_loads['m'] = m
    df_DLB_fatigue_loads['value'] = value
    
    df_DLB_fatigue_loads.to_excel(path, index=False)

def plot_DLB_extreme_loads(DLB_extreme_loads, folder, extension='.png'):
    """
    Plot the DLB extreme loads and save them to a given folder.

    Parameters
    ----------
    DLB_extreme_loads : xarray.DataArray ('sensor_name', 'angle', '(value, dlc, group)')
        DataArray containing the extreme loads of the DLB. It matches the
        output from get_DLB_extreme_loads
    folder : str
        Path to the folder where the plots will be saved
    extension: str, optional
        Extension of the plot files. The default is '.png'.

    Returns
    -------
    None.

    """
    angles = DLB_extreme_loads.coords['angle'].values
    for i in range(DLB_extreme_loads.shape[0]):
        x = [float(DLB_extreme_loads[i, j, 0])*np.cos(np.deg2rad(angles[j]))
             for j in range(len(angles))]
        y = [float(DLB_extreme_loads[i, j, 0])*np.sin(np.deg2rad(angles[j]))
             for j in range(len(angles))]
        DLC = [DLB_extreme_loads[i, j, 1].values[()] for j in range(len(angles))]
        plt.scatter(x, y)
        [plt.plot([0, x[k]], [0, y[k]], color='black') for k in range(len(x))]
        plt.xlabel('Mx')
        plt.ylabel('My')
        plt.title(DLB_extreme_loads[i].coords['sensor_name'].values[()])
        plt.axhline(0, color='black',linewidth=1)  
        plt.axvline(0, color='black',linewidth=1)  
        plt.xlim(-max(abs(min(x)), abs(max(x)))*1.2, max(abs(min(x)), abs(max(x)))*1.2)
        plt.ylim(-max(abs(min(y)), abs(max(y)))*1.2, max(abs(min(y)), abs(max(y)))*1.2)
        for j in range(len(x)):
            plt.annotate(DLC[j], (x[j], y[j]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.savefig(os.path.join(folder, 'Extreme_' + DLB_extreme_loads[i].coords['sensor_name'].values[()] + extension))
        plt.show()
        
def plot_DLB_fatigue_loads(DLB_fatigue_loads, folder, extension='.png'):
    """
    Plot the DLB extreme loads and save them to a given folder.

    Parameters
    ----------
    DLB_fatigue_loads : xarray.DataArray ('sensor_name', 'angle', 'm')
        DataArray containing the fatigue loads of the DLB. It matches the
        output from get_DLB_fatigue_loads
    folder : str
        Path to the folder where the plots will be saved
    extension: str, optional
        Extension of the plot files. The default is '.png'.

    Returns
    -------
    None.

    """
    m_list = DLB_fatigue_loads.coords['m'].values
    angles = DLB_fatigue_loads.coords['angle'].values
    for i in range(DLB_fatigue_loads.shape[0]):
        for j in range(len(m_list)):
            x = [float(DLB_fatigue_loads[i, k, j])*np.cos(np.deg2rad(angles[k]))
                 for k in range(len(angles))]
            y = [float(DLB_fatigue_loads[i, k, j])*np.sin(np.deg2rad(angles[k]))
                 for k in range(len(angles))]
            plt.scatter(x, y, label='m = ' + str(m_list[j]))
            [plt.plot([0, x[k]], [0, y[k]], color='black') for k in range(len(angles))]
        plt.xlabel('Mx')
        plt.ylabel('My')
        plt.title(DLB_fatigue_loads[i].coords['sensor_name'].values[()])
        plt.axhline(0, color='black',linewidth=1)
        plt.axvline(0, color='black',linewidth=1)
        plt.xlim(-max(abs(min(x)), abs(max(x)))*1.2, max(abs(min(x)), abs(max(x)))*1.2)
        plt.ylim(-max(abs(min(y)), abs(max(y)))*1.2, max(abs(min(y)), abs(max(y)))*1.2)
        plt.legend()
        plt.savefig(os.path.join(folder, 'Fatigue_' + DLB_fatigue_loads[i].coords['sensor_name'].values[()] + extension))
        plt.show()