# -*- coding: utf-8 -*-




constants = {'simulation.time_stop':100}


variables = {'wind.wsp': [4, 6, 8],
             'wind.tint': [0.1, 0.15, 0.2]}



functions = {'Name': lambda x: 'sim_wsp' + str(x['wind.wsp']) + '_ti' + str(x['wind.tint'])}



