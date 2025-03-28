"""
scripter file
Created on Thu Jul 18 17:38:17 2024

@author: leo

Generates several datasets in sequence and saves them into a file.
"""

import scan_parameterspace_funcs as fcs
import scan_parameterspace as spr
import pandas as pd
import numpy as np
from pathlib import Path
from time import time

Number_of_datasets = 1
Numer_of_points = 2e5

if fcs.alignment:
    strga = 'A'
else:
    strg = str(fcs.non_alignment_max)
    strga = 'NA'+strg
    
if fcs.small_l5:
    strgl5 = '331lk'
else:
    strgl5 = ''

set_dir = Path('./data_'+'GTHDM'+fcs.THDM_type+strgl5+'-'+strga+'/')
set_dir.mkdir(parents=True, exist_ok=True)

print('Generating points for '+spr.latex_model+' in')
print('alignment' if fcs.alignment else 'non-alignment '+strg)
print('with small l5' if fcs.small_l5 else 'varying l5')

path_files = [Path('./'+set_dir.parts[0]+'/GTHDM'+fcs.THDM_type+strgl5+'-'+strga+'-Theo_PDG.csv'),Path('./'+set_dir.parts[0]+'/GTHDM'+fcs.THDM_type+strgl5+'-'+strga+'-STU_PDG.csv'),Path('./'+set_dir.parts[0]+'/GTHDM'+fcs.THDM_type+strgl5+'-'+strga+'-Collid_PDG.csv'),Path('./'+set_dir.parts[0]+'/GTHDM'+fcs.THDM_type+strgl5+'-'+strga+'-BSG_PDG.csv'),Path('./'+set_dir.parts[0]+'/GTHDM'+fcs.THDM_type+strgl5+'-'+strga+'-PU_PDG.csv')]

strt = time()
for i in range(Number_of_datasets):
    res = spr.main_module(Numer_of_points)
    header = [not path_files[0].exists(),not path_files[1].exists(),not path_files[2].exists(),not path_files[3].exists(),not path_files[4].exists()]
    res[0].to_csv(path_files[0],mode='a',header=header[0],index=False)
    res[1].to_csv(path_files[1],mode='a',header=header[1],index=False)
    res[2].to_csv(path_files[2],mode='a',header=header[2],index=False)
    res[3].to_csv(path_files[3],mode='a',header=header[3],index=False)
    res[4].to_csv(path_files[4],mode='a',header=header[4],index=False)

end = time()
print("Total time: %.2f days" %((end-strt)/(3600*24)))