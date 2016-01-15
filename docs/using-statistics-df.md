How to use the Statistics DataFrame
===================================


Introduction
------------

The statistical data of your post-processed load cases are saved in the HDF
format. You can use Pandas to retrieve and organize that data. Pandas organizes
the data in a DataFrame, and the library is powerful, comprehensive and requires
some learning. There are extensive resources out in the wild that can will help
you getting started:

* [list](http://pandas.pydata.org/pandas-docs/stable/tutorials.html)
of good tutorials can be found in the Pandas
[documentation](http://pandas.pydata.org/pandas-docs/version/0.16.2/).
* short and simple
[tutorial](https://github.com/DTUWindEnergy/Python4WindEnergy/blob/master/lesson%204/pandas.ipynb)
as used for the Python 4 Wind Energy course


The data is organized in simple 2-dimensional table. However, since the statistics
of each channel is included for multiple simulations, the data set is actually
3-dimensional. As an example, this is how a table could like:

```
   [case_id]  [channel name]  [mean]  [std]    [windspeed]
       sim_1          pitch        0      1              8
       sim_1            rpm        1      7              8
       sim_2          pitch        2      9              9
       sim_2            rpm        3      2              9
       sim_3          pitch        0      1              7
```

Each row is a channel of a certain simulation, and the columns represent the
following:

* a tag from the master file and the corresponding value for the given simulation
* the channel name, description, units and unique identifier
* the statistical parameters of the given channel


Load the statistics as a pandas DataFrame
-----------------------------------------

Pandas has some very powerful functions that will help analysing large and
complex DataFrames. The documentation is extensive and is supplemented with
various tutorials. You can use
[10 Minutes to pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html)
as a first introduction.

Loading the pandas DataFrame table works as follows:

```python
import pandas as pd
df = pd.read_hdf('path/to/file/name.h5', 'table')
```

Some tips for inspecting the data:

```python
import numpy as np

# Check the available data columns:
for colname in sorted(df.keys()):
    print colname

# list all available channels:
print np.sort(df['channel'].unique())

# list the different load cases
df['[Case folder]'].unique()
```


Reduce memory footprint using categoricals
------------------------------------------

When the DataFrame is consuming too much memory, you can try to reduce its
size by using categoricals. A more extensive introduction to categoricals can be
found
[here](http://pandas.pydata.org/pandas-docs/stable/faq.html#dataframe-memory-usage)
and [here](http://matthewrocklin.com/blog/work/2015/06/18/Categoricals/).
The basic idea is to replace all string values with an integer,
and have an index that maps the string value to the index. This trick only works
when you have long strings that occur multiple times throughout your data set.

The following example shows how you can use categoricals to reduce the memory
usage of a pandas DataFrame:

```python
# load a certain DataFrame
df = pd.read_hdf(fname, 'table')
# Return the total estimated memory usage
print '%10.02f MB' % (df.memory_usage(index=True).sum()/1024.0/1024.0)
# the data type of column that contains strings is called object
# convert objects to categories to reduce memory consumption
for column_name, column_dtype in df.dtypes.iteritems():
    # applying categoricals mostly makes sense for objects, we ignore all others
    if column_dtype.name == 'object':
        df[column_name] = df[column_name].astype('category')
print '%10.02f MB' % (df.memory_usage(index=True).sum()/1024.0/1024.0)
```

Python has a garbage collector working in the background that deletes
un-referenced objects. In some cases it might help to actively trigger the
garbage collector as follows, in an attempt to free up memory during a run of
a script that is almost flooding the memory:

```python
import gc
gc.collect()
```

Load a DataFrame that is too big for memory in chunks
-----------------------------------------------------

When a DataFrame is too big to load into memory at once, and you already
compressed your data using categoricals (as explained above), you can read
the DataFrame one chunk at the time. A chunk is a selection of rows. For
example, you can read 1000 rows at the time by setting ```chunksize=1000```
when calling ```pd.read_hdf()```. For example:

```python
# load a large DataFrame in chunks of 1000 rows
for df_chunk in pd.read_hdf(fname, 'table', chunksize=1000):
    print 'DataFrame chunk contains %i rows' % (len(df_chunk))
```

We will read a large DataFrame as chunks into memory, and select only those
rows who belong to dlc12:

```python
# only select one DLC, and place them in one DataFrame. If the data
# containing one DLC is still to big for memory, this approach will fail

# create an empty DataFrame, here we collect the results we want
df_dlc12 = pd.DataFrame()
for df_chunk in pd.read_hdf(fname, 'table', chunksize=1000):
    # organize the chunk: all rows for which [Case folder] is the same
    # in a single group. Each group is now a DataFrame for which
    # [Case folder] has the same value.
    for group_name, group_df in df_chunk.groupby(df_chunk['[Case folder]']):
        # if we have the group with dlc12, save them for later
        if group_name == 'dlc12_iec61400-1ed3':
            df_dlc12 = pd.concat([df_dlc12, group_df])#, ignore_index=True)
```

Plot wind speed vs rotor speed
------------------------------

```python
# select the channels of interest
for group_name, group_df in df_dlc12.groupby(df_dlc12['channel']):
    # iterate over all channel groups, but only do something with the channels
    # we are interested in
    if group_name == 'Omega':
        # we save the case_id tag, the mean value of channel Omega
        df_rpm = group_df[['[case_id]', 'mean']].copy()
        # note we made a copy because we will change the DataFrame in the next line
        # rename the column mean to something more useful
        df_rpm.rename(columns={'mean': 'Omega-mean'}, inplace=True)
    elif group_name == 'windspeed-global-Vy-0.00-0.00--127.00':
        # we save the case_id tag, the mean value of channel wind, and the 
        # value of the Windspeed tag
        df_wind = group_df[['[case_id]', 'mean', '[Windspeed]']].copy()
        # note we made a copy because we will change the DataFrame in the next line
        # rename the mean of the wind channel to something more useful
        df_wind.rename(columns={'mean': 'wind-mean'}, inplace=True)

# join both results on the case_id value so the mean RPM and mean wind speed
# are referring to the same simulation/case_id.
df_res = pd.merge(df_wind, df_rpm, on='[case_id]', how='inner')

# now we can plot RPM vs wind speed
from matplotlib import pyplot as plt
plt.plot(df_res['wind-mean'].values, df_res['Omega-mean'].values, '*')
```

