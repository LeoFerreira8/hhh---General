#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:00:12 2024

@author: leo
"""

import pandas as pd
import fileinput
from subprocess import run
from contextlib import chdir
import datetime
import scan_parameterspace_funcs as fcs

Spheno_path = '/home/leo/Documents/Unicamp/HEPTools/SPheno/'
keep_log = False

def write_spheno_LHA(par_list):
    '''Input parameters for SPheno for THDM. The parameters are lambda1,lambda2,lambda3,lambda4,lambda5,m12 and tan(beta).'''
    
    file = fileinput.input(Spheno_path+"LesHouches.in.THDMIInoZ2",inplace=True)
    block = 0
    
    # The block for input in SPheno looks like:
    # Block MINPAR      # Input parameters 
    # 1   1.0000000E-01    # Lambda1Input
    # 2   1.3000000E-01    # Lambda2Input
    # 3   1.1000000E+00    # Lambda3Input
    # 4   -5.0000000E-01    # Lambda4Input
    # 5   5.0000000E-01    # Lambda5Input
    # 6   1.0000000E-02    # Lambda6Input
    # 7   1.0000000E-02    # Lambda7Input
    # 9   4.0000000E+04    # M12input
    # 10   5.0000000E+01    # TanBeta
    
    lines_sp = [
    ' 1   %e    # Lambda1Input',
    ' 2   %e    # Lambda2Input',
    ' 3   %e    # Lambda3Input',
    ' 4   %e    # Lambda4Input',
    ' 5   %e    # Lambda5Input',
    ' 6   %e    # Lambda6Input',
    ' 7   %e    # Lambda7Input',
    ' 9   %e    # M12input',
    ' 10  %e    # TanBeta']
    
    line_no = 0
    #list_test = ['a','b','c','d','e','f','g']
    
    for line in file:
        if block:
            print(lines_sp[line_no] %(par_list[line_no]))
            #print(list_test[line_no],end='')
            line_no+=1
            if line_no == (len(lines_sp)):
                block = 0
        elif 'Block MINPAR      # Input parameters' in line:
            block = 1 #Finds the block / skips the first line
            print(line, end='')
        else:
            print(line, end='')
            
    fileinput.close()
    
    return 0
            
def execute_spheno():
    with chdir(Spheno_path):
        first_call=run(["bin/SPhenoTHDMIInoZ2","LesHouches.in.THDMIInoZ2"],capture_output=True)
        
        if keep_log:
            fil = open("./logs/SPhen_log"+str(datetime.datetime.now()),'w')
            
            fil.write(first_call.stdout.decode())
            
            fil.close()
        
        if ("Finished!" in first_call.stdout.decode()):
            return 1
        else:
            return 0
        
def read_spheno_obs():
    f = open(Spheno_path+"SPheno.spc.THDMIInoZ2",'r')
    text = f.readlines()[27:]
    dt_in=pd.DataFrame()
    dt_el=pd.DataFrame()
    dt_fl=pd.DataFrame()
    dt_fll=pd.DataFrame()
    
    # Get input parameters
    block = 0
    block_name = 'MINPAR'
    for line in text:
        if block_name in line:
            f.readlines()
            block = 1
        elif 'Block' in line and block_name not in line:
            block = 0
        elif block==1:
            proxy = pd.Series(line)
            proxy = proxy.str.split(n=2,expand=True).T
            for series_name, series in proxy.loc[2].items():
                proxy.at[2,series_name]=series[2:].strip()
            proxy = proxy.rename(columns=proxy.loc[2])
            proxy = proxy.drop(2)
            dt_in=pd.concat([dt_in,proxy],axis=1)
            
    # Get low energy observables
    block = 0
    block_name = 'SPhenoLowEnergy'
    for line in text:
        if block_name in line:
            f.readlines()
            block = 1
        elif 'Block' in line and block_name not in line:
            block = 0
        elif block==1:
            proxy = pd.Series(line)
            proxy = proxy.str.split(n=2,expand=True).T
            for series_name, series in proxy.loc[2].items():
                proxy.at[2,series_name]=series[2:].strip()
            proxy = proxy.rename(columns=proxy.loc[2])
            proxy = proxy.drop(2)
            dt_el=pd.concat([dt_el,proxy],axis=1)
            
    # Get flavor observables
    block = 0
    block_name = 'FlavorKitQFV'
    for line in text:
        if block_name in line:
            f.readlines()
            block = 1
        elif 'Block' in line and block_name not in line:
            block = 0
        elif block==1:
            proxy = pd.Series(line)
            proxy = proxy.str.split(n=2,expand=True).T
            for series_name, series in proxy.loc[2].items():
                proxy.at[2,series_name]=series[2:].strip()
            proxy = proxy.rename(columns=proxy.loc[2])
            proxy = proxy.drop(2)
            dt_fl=pd.concat([dt_fl,proxy],axis=1)
            
    # Get flavor observables
    block = 0
    block_name = 'FlavorKitLFV'
    for line in text:
        if block_name in line:
            f.readlines()
            block = 1
        elif 'Block' in line and block_name not in line:
            block = 0
        elif block==1:
            proxy = pd.Series(line)
            proxy = proxy.str.split(n=2,expand=True).T
            for series_name, series in proxy.loc[2].items():
                proxy.at[2,series_name]=series[2:].strip()
            proxy = proxy.rename(columns=proxy.loc[2])
            proxy = proxy.drop(2)
            dt_fll=pd.concat([dt_fll,proxy],axis=1)
            
    f.close()
    
    df = pd.concat([dt_in,dt_el,dt_fl,dt_fll],axis=1)
    df = df.replace('E', 'e', regex=True).replace(',', '.', regex=True)
    # convert notation to the one pandas allows
    df = df.apply(pd.to_numeric, args=('coerce',))
    
    return df
    