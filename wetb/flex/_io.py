'''
Created on 11. apr. 2017

@author: mmpe
'''
import struct
import numpy as np
import os


def load(filename, name_stop=8, dtype=float):
    """
    Parameters
    ----------
    - filename
    - name_stop : int or str
        if int: Number of characters for name
        if str: name-description delimiter, e.g. " "
    - dtype
    """
    if isinstance(filename, str):
        fid = open(filename, 'rb')
    elif hasattr(filename, "name"):
        fid = filename
        filename = fid.name
    try:
        _ = struct.unpack('i', fid.read(4))
        _ = struct.unpack('i', fid.read(4))
        title = fid.read(60).strip()

        _ = struct.unpack('i', fid.read(4))
        _ = struct.unpack('i', fid.read(4))
        no_sensors = struct.unpack('i', fid.read(4))[0]

        sensor_numbers = [struct.unpack('i', fid.read(4))[0] for _ in range(no_sensors)]
        _ = struct.unpack('i', fid.read(4))
        _ = struct.unpack('i', fid.read(4))
        time_start = struct.unpack('f', fid.read(4))[0]
        time_step = struct.unpack('f', fid.read(4))[0]
        scale_factors = np.array([struct.unpack('f', fid.read(4))[0] for _ in range(no_sensors)], dtype=np.double)
        data = np.fromstring(fid.read(), 'int16').astype(dtype)
    finally:
        fid.close()

#     if title.isalnum():
#         self.set_info(title, "", self.filename)
#     else:
#         self.set_info(os.path.basename(self.name), "", self.name)
    try:
        data = data.reshape(len(data) // no_sensors, no_sensors)
    except ValueError:
        raise ValueError(
            "The number of data values (%d) is not divisible by the number of sensors (%d)" %
            (len(data), no_sensors))

    for i in range(data.shape[1]):
        data[:, i] *= scale_factors[i]
    no_scans = data.shape[0]

    # Load sensor file if exists

    sensor_filename = os.path.join(os.path.dirname(filename), "sensor")
    sensor_info = {info[0]: info[1:] for info in read_sensor_info(sensor_filename, name_stop)}

    # set gain and offset for "Time"
    gains = []
    offsets = []
    names, units, descriptions = [], [], []

    for sensor_nr in sensor_numbers:

        name, unit, description, gain, offset = sensor_info.get(sensor_nr, ["Attribute %d" % sensor_nr, '-', '', 1, 0])
        if sensor_nr == 1 and name == "Time" and unit == "s":
            data = data[:, 1:]
            continue
        names.append(name)
        units.append(unit)
        descriptions.append(description)
        gains.append(gain)
        offsets.append(offset)

    time = np.arange(time_start, time_start + data.shape[0] * time_step, time_step).reshape((no_scans, 1))
    #data = np.concatenate((time, data), axis=1)
    #gains = np.r_[1,gains]
    #offsets = np.r_[0,offsets]
    # self[:]*=self.gains
    # self[:]+=self.offsets
    info = {"name": title,
            "description": filename,
            "attribute_names": names,
            "attribute_units": units,
            "attribute_descriptions": descriptions}
    return time, data, info


def read_sensor_info(sensor_file, name_stop=8):
    """
    Parameters
    ----------
    - sensor_file
    - name_stop : int or str
        if int: Number of characters for name
        if str: name-description delimiter, e.g. " "
    """

    if hasattr(sensor_file, 'readlines'):
        sensor_info_lines = sensor_file.readlines()[2:]
    else:
        with open(sensor_file, encoding="utf-8") as fid:
            sensor_info_lines = fid.readlines()[2:]
    sensor_info = []
    for line in sensor_info_lines:
        # while "  " in line:
        #    line = line.replace("  "," ")
        line = line.strip().split()
        nr = int(line[0])
        gain = float(line[1])
        offset = float(line[2])
        unit = line[5]
        if isinstance(name_stop, int):
            name_desc = " ".join(line[6:])
            name = name_desc[:name_stop].strip()
            description = name_desc[name_stop:].strip()
        elif isinstance(name_stop, str):
            name_desc = (" ".join(line[6:])).split(name_stop)
            name = name_desc[0].strip()
            description = name_stop.join(name_desc[1:]).strip()

        sensor_info.append((nr, name, unit, description, gain, offset))
    return sensor_info

# def save(dataset, filename):
#     ds = dataset
#     # Write int data file
#     f = open(filename, 'wb')
#     time_att = ds.basis_attribute()
#     sensors = [s for s in ds.attributes() if s is not time_att]
#
#     if isinstance(ds, FLEX4Dataset):
#         data = ds[:]  # (ds[:]-ds.offsets)/ds.gains
#     else:
#         data = ds[:]
#     if time_att.index != -1:  # time_att may be "Index" with index=-1 if "Time" not exists
#         data = np.delete(data, time_att.index, axis=1)
#     f.write(struct.pack('ii', 0, 0))  # 2x empty int
#     title = ("%-60s" % str(ds.name)).encode()
#     f.write(struct.pack('60s', title))  # title
#     f.write(struct.pack('ii', 0, 0))  # 2x empty int
#     ns = len(sensors)
#     f.write(struct.pack('i', ns))
#     f.write(struct.pack('i' * ns, *range(1, ns + 1)))  # sensor number
#     f.write(struct.pack('ii', 0, 0))  # 2x empty int
#     time = ds.basis_attribute()
#     f.write(struct.pack('ff', time[0], time[1] - time[0]))  # start time and time step
#
#     scale_factors = np.max(np.abs(data), 0) / 32000
#     f.write(struct.pack('f' * len(scale_factors), *scale_factors))
#     # avoid dividing by zero
#     not0 = np.where(scale_factors != 0)
#     data[:, not0] /= scale_factors[not0]
#     #flatten and round
#     data = np.round(data.flatten()).astype(np.int16)
#     f.write(struct.pack('h' * len(data), *data.tolist()))
#     f.close()
#
#     # write sensor file
#     f = open(os.path.join(os.path.dirname(filename), 'sensor'), 'w')
#     f.write("Sensor list for %s\n" % filename)
#     f.write(" No   forst  offset  korr. c  Volt    Unit   Navn    Beskrivelse------------\n")
#     sensorlineformat = "%3s  %.3f   %.3f      1.00     0.00 %7s %-8s %s\n"
#
#     if isinstance(ds, FLEX4Dataset):
#         gains = np.r_[ds.gains[1:], np.ones(ds.shape[1] - len(ds.gains))]
#         offsets = np.r_[ds.offsets[1:], np.zeros(ds.shape[1] - len(ds.offsets))]
#         sensorlines = [sensorlineformat % ((nr + 1), gain, offset, att.unit[:7], att.name.replace(" ", "_")[:8], att.description[:512]) for nr, att, gain, offset in zip(range(ns), sensors, gains, offsets)]
#     else:
#         sensorlines = [sensorlineformat % ((nr + 1), 1, 0, att.unit[:7], att.name.replace(" ", "_")[:8], att.description[:512]) for nr, att in enumerate(sensors)]
#     f.writelines(sensorlines)
#     f.close()
