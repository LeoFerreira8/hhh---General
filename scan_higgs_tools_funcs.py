#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:18:31 2024

@author: leo
"""

import Higgs.predictions as HP
import Higgs.bounds as HB
import Higgs.signals as HS
import Higgs.tools.Input as hinput
import scan_SPheno_funcs as SPfcs
from contextlib import chdir
import datetime
import pandas as pd
import scan_parameterspace_funcs as fcs

def write_or_update(file_path, line_to_write):
    # A dictionary to store lines and their counters
    lines_dict = {}

    try:
        # Read the file content
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Populate the dictionary with current lines and their counters
        for line in lines:
            if ':' in line:
                content, count = line.rsplit(':', 1)
                lines_dict[content.strip()] = int(count.strip())
            else:
                lines_dict[line.strip()] = 1  # Initialize a count of 1 if no counter

        # Update the count for the line_to_write
        if line_to_write in lines_dict:
            lines_dict[line_to_write] += 1
        else:
            lines_dict[line_to_write] = 1  # Add the new line with a count of 1

        # Write the updated lines back to the file
        with open(file_path, 'w') as file:
            for line, count in lines_dict.items():
                file.write(f"{line}: {count}\n")

    except FileNotFoundError:
        # If the file does not exist, create it with the line and counter
        with open(file_path, 'w') as file:
            file.write(f"{line_to_write}: 1\n")

keep_log = False
Higgs_tools_database_path = "/home/leo/Documents/Unicamp/HEPTools/higgstools/Database/"

bounds = HB.Bounds(Higgs_tools_database_path+"hbdataset/") # load HB dataset
signals = HS.Signals(Higgs_tools_database_path+"hsdataset/") # load HS dataset

neutralIds = [25,35,36]
chargedIds = [37]
neutralIdStrings = [str(id) for id in neutralIds]
chargedIdStrings = [str(id) for id in chargedIds]

def Higgs_tools_scan():
    dc = hinput.readHB5SLHA(SPfcs.Spheno_path+"SPheno.spc.THDMIInoZ2", neutralIds, chargedIds)
    pred = hinput.predictionsFromDict(dc, neutralIdStrings, chargedIdStrings, [])
    h = pred.particle('25')

    # evaluate HiggsBounds
    hbresult = bounds(pred)
    
    # evaluate HiggsSignals
    chisq = signals(pred)
    
    with chdir(Higgs_tools_database_path+'Logs'):
        
        if keep_log and not bool(hbresult):
            fil = open("Higgs_tools_log"+str(datetime.datetime.now()),'w')
            
            fil.write(str(hbresult.allowed)+'\n')
            fil.write('h: '+str(hbresult.selectedLimits['25'].limit())+'\n' if '25' in hbresult.selectedLimits else '\n')
            fil.write('H: '+str(hbresult.selectedLimits['35'].limit())+'\n' if '35' in hbresult.selectedLimits else '\n')
            fil.write('A: '+str(hbresult.selectedLimits['36'].limit())+'\n' if '36' in hbresult.selectedLimits else '\n')
            fil.write('Hpm: '+str(hbresult.selectedLimits['37'].limit())+'\n' if '37' in hbresult.selectedLimits else '\n')
            fil.write("HiggsSignals chisq: %f" %chisq)
            
            fil.close()
            write_or_update('/home/leo/Documents/Unicamp/Doutorado/3h/ColBounds.txt', 'h: '+str(hbresult.selectedLimits['25'].limit()) if '25' in hbresult.selectedLimits else '')
            write_or_update('/home/leo/Documents/Unicamp/Doutorado/3h/ColBounds.txt', 'H: '+str(hbresult.selectedLimits['35'].limit()) if '35' in hbresult.selectedLimits else '')
            write_or_update('/home/leo/Documents/Unicamp/Doutorado/3h/ColBounds.txt', 'A: '+str(hbresult.selectedLimits['36'].limit()) if '36' in hbresult.selectedLimits else '')
            write_or_update('/home/leo/Documents/Unicamp/Doutorado/3h/ColBounds.txt', 'Hpm: '+str(hbresult.selectedLimits['37'].limit()) if '37' in hbresult.selectedLimits else '')
    
    outpt=pd.DataFrame({'HiggsB': bool(hbresult), 'HiggsS': chisq},index=[1])

    return outpt