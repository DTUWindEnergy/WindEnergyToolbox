#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 11:07:50 2021

@author: dave
"""

# from os.path import join as pjoin

import numpy as np
from lxml import objectify#, etree)

# from matplotlib import pyplot as plt
# import pandas as pd

# import wetb
from wetb.prepost import misc
# from wetb.hawc2 import (HTCFile, AEFile, PCFile, StFile)


# TODO: this dictionary utility could go to a dict sub-class in misc?
def find_next_unique_key(d, key):
    k = 1
    while True:
        k += 1
        if key + f'__{k}' not in d:
            key = key + f'__{k}'
            return key


# TODO: this dictionary utility could go to a dict sub-class in misc?
def add_new_unique_key(key, values, d):
    """
    Add key:values to dictionary d. If key already occurs, append __x
    where x is the number of previous occurances of key(__x).

    Parameters
    ----------
    key : str
        DESCRIPTION.
    values : list
        DESCRIPTION.
    d : dict
        Dictionary.

    Returns
    -------
    key : str
        DESCRIPTION.
    d : dict
        DESCRIPTION.

    """
    if key in d:
        key = find_next_unique_key(d, key)
    d[key] = [values]
    return key, d


class ReadBladedProject:

    def __init__(self, fname):

        with open(fname, encoding='utf-8') as fobj:
            xml_str = fobj.read().encode('utf-8')
        self.bd, self.xmlroot = self.parse_bladeddata(xml_str)

        self.set_attr_and_check()

        # some things are just a little different
        # TMASS has a list of materials and their properties
        # get rid of the quotes
        # tmp = [el.replace("'", '') for el in self.bd['TMASS']['MATERIAL'][0]]
        # self.bd['TMASS']['MATERIAL'] = tmp
        unique_mat = set(self.get_key('TMASS', 'MATERIAL').flatten().tolist())
        self.tow_mat_prop = {k:self.get_key('TMASS', k) for k in unique_mat}
        # material_props = {}
        # for k in unique_mat:
        #     material_props[k.replace("'", '')] = self.get_key('TMASS', k)

    def parse_bladeddata(self, xml_str):
        """
        The XML field BladedData contains what seems like the main core input
        data for BLADED, and formatted in some structured way.

        Parameters
        ----------
        xml_str : TYPE
            DESCRIPTION.

        Returns
        -------
        bd : TYPE
            DESCRIPTION.
        xmlroot : TYPE
            DESCRIPTION.

        """

        # root = etree.fromstring(xml)
        # elems = root.getchildren()
        # bladeddata = elems[1].text

        xmlroot = objectify.fromstring(xml_str)

        # the non-xml formatted BLADED model is situated in a non-xml field
        # called BladedData
        bd = {}
        mstart = None
        for i, line in enumerate(xmlroot.BladedData.text.split('\n')):

            # TODO: values embedded in double quotes (") can contain entire
            # MSTART/MEND sub-sections (so embedded sections)

            # split replace tabs with spaces, split on spaces, remove empty
            linels = misc.remove_items(line.replace('\t', ' ').split(' '), '')
            # commas can also be separators, in addition to spaces
            linels2 = []
            for k in linels:
                linels2.extend(k.split(','))
            linels = misc.remove_items(linels2, '')

            # ignore empty lines
            if len(linels) < 1:
                continue

            # entries can be numbers if the previous key contains multiple data points
            try:
                float(linels[0])
                el0isnum = True
            except ValueError:
                el0isnum = False

            # start of a sub-section that contains (non-unique) keys as well
            if linels[0].upper().startswith('MSTART'):
                mtag = linels[-1]
                mstart = {}

            # at the end of the sub-section add the sub-section to the main dict
            elif linels[0].upper().startswith('MEND'):
                # FIXME: implement MSTART sections embedded in double quoted values
                try:
                    # if the section key is not unique, make it so by appending __X
                    if mtag in bd:
                        mtag = find_next_unique_key(bd, mtag)
                    bd[mtag] = mstart
                except UnboundLocalError:
                    print('warning: ignored embedded mstart/mend section')
                    print(f'at line: {i+1}')
                mstart = None

            # if we are under a sub-section
            elif mstart is not None:
                # if the line contains a keyword
                if not el0isnum:
                    tag, mstart = add_new_unique_key(linels[0], linels[1:], mstart)
                # line is datapoint that needs to be added to key that occured before
                else:
                    mstart[tag].append(linels)

            # add numerical values to key that occured before
            elif el0isnum:
                 bd[tag].append(linels)

            else:
                tag, bd = add_new_unique_key(linels[0], linels[1:], bd)

        return bd, xmlroot

    def get_key(self, key1, key2=False):
        """
        Get key from the BladedData CDATA section and format to int32 or
        float32 numpy arrays if possible withouth precision loss.

        Parameters
        ----------
        key1 : str
            DESCRIPTION.
        key2 : str, optional
            DESCRIPTION. The default is False.


        Returns
        -------
        numpy.array
            Values from key1/key2 formatted as a numpy array. Converted to
            numpy.int32, numpy.float32 if possible withouth precision loss,
            otherwise an object array is returned.

        """

        if key1 not in self.bd:
            raise KeyError(f'{key1} not found in BLADED file.')

        if key2 is not False:
            if key2 not in self.bd[key1]:
                raise KeyError(f'{key2} not found in MSTART {key1} of BLADED file.')
            data = self.bd[key1][key2]
        else:
            data = self.bd[key1]

        # in case we defined a mstart key
        if isinstance(data, dict):
            return data

        # i ,j = len(data), len(data[0])

        # FIXME: this is a very expensive way of converting it, but it might
        # not matter at all since very little model data is actually considered
        data_arr = np.array(data)
        try:
            data_arr = data_arr.astype(np.int32)
        except ValueError:
            try:
                data_arr = data_arr.astype(np.float32)
            except ValueError:
                pass

        return data_arr

        # return np.array(data, dtype=np.float32)

        # if isinstance(data[0], list) and len(data[0]) == 1:
        #     data = float(data[0])
        # if isinstance(data[0], list) and len(data[0]) > 1:
        #     data_arr = np.array(data, dtype=np.float32)

    def set_attr_and_check(self):
        """Check that BGEOMMB, BMASSMB, BSTIFFMB has indeed the same node
        repated twice every time.
        """

        # self.bd['BGEOMMB'].keys(), but only those relevant
        keysg = ['RJ', 'DIST', 'REF_X', 'REF_Y', 'CHORD', 'TWIST', 'CE_X',
                 'CE_Y', 'BTHICK', 'FOIL', 'MOVING']
        nbe = self.get_key('BGEOMMB', 'NBE')[0,0]

        # self.bd['BMASSMB'].keys(), but only those relevant
        keysm = ['CM_X', 'CM_Y', 'MASS', 'SINER', 'RGRATIO', 'BETA_M']

        # self.bd['BSTIFFMB'].keys(), but only those relevant
        keyss = ['EIFLAP', 'EIEDGE', 'BETA_S', 'GJ', 'CS_X', 'CS_Y', 'GAFLAP',
                 'GAEDGE']

        mkeys = ['BGEOMMB', 'BMASSMB', 'BSTIFFMB']
        for mkey, keys in zip(mkeys, [keysg, keysm, keyss]):
            for key in keys:
                res = self.get_key(mkey, key)
                try:
                    assert np.allclose(res[0,0::2], res[0,1::2])
                except TypeError:
                    # allclose doesn't make sense for text arrays
                    assert np.compare_chararrays(res[0,0::2], res[0,1::2],
                                                 '==', True).all()
                assert res.shape[1]==nbe
                if hasattr(self, key.lower()):
                    raise UserWarning(key, 'already exists')
                setattr(self, key.lower(), res[0,0::2])

    def print_xml_tree(self, fname):
        """For discovery purposes: print full tree + values/text


        Parameters
        ----------
        fname : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # def print_rec(root):
        #     # if hasattr(root, 'getparent'):
        #     #     print(root.getparent().tag.title, end='.')
        #     #     print_rec(root.getparent())
        #     for el in root.getchildren():
        #         print(print_rec(el))

        # def print_root_tree(root):
        #     root.getparent()
        #     print()

        # tree = etree.fromstring(xml_str)
        # els = tree.xpath('/')
        # for el in els:
        #     print(el)

        # tree = etree.fromstring(xml_str)

        # xmlroot = objectify.fromstring(xml_str)
        # with open(fname+'.structure', 'w') as f:
        #     for line in xmlroot.descendantpaths():
        #         f.write(line + '\n')

        # Recursive XML parsing python using ElementTree
        # https://stackoverflow.com/q/28194703/3156685

        roottree = self.xmlroot.getroottree()
        def print_tree_recursive(root):
            # print(roottree.getpath(root), end=' : ')
            # print(root.tag.title())
            f.write(roottree.getpath(root) + ' : ')
            f.writelines(root.values())
            if root.text is not None:
                f.write(' : ' + root.text)
            f.write('\n')
            for elem in root.getchildren():
                print_tree_recursive(elem)
        with open(fname+'.structure.value', 'w') as f:
            for el in self.xmlroot.getchildren():
                if el.tag.title()!='Bladeddata':
                    print_tree_recursive(el)

    # def get_frequencies(self):
    #     """
    #     """

    #     blades = self.xmlroot.NewGUI.Turbine.Blades.FlexibilityModel

    #     blades.Settings.ModesWithDampingDefined
    #     blades.PartModes # with lots of BladeModeContainer
    #     blades.WholeBladeModes
    #     blades.WholeBladeModes.WholeBladeMode

    #     blades.WholeBladeModes.WholeBladeMode.Name
    #     blades.WholeBladeModes.WholeBladeMode.Frequency
    #     blades.WholeBladeModes.WholeBladeMode.Damping
    #     blades.WholeBladeModes.WholeBladeMode.Components
    #     len(blades.WholeBladeModes.WholeBladeMode.Components.getchildren())
    #     # 60 elements

    #     for wholeblademode in blades.WholeBladeModes.iterchildren():
    #         print(wholeblademode.Name)
    #         print(wholeblademode.Frequency, wholeblademode.Damping)

    #     tower = self.xmlroot.NewGUI.Turbine.SupportStructure.FlexibilityModel
    #     for towermode in tower.Modes.getchildren():
    #         print(towermode.Description)
    #         towermode.Frequency
    #         towermode.Damping
