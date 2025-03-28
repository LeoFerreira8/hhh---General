#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:46:43 2024

@author: leo
"""

import numpy as np
import scan_parameterspace_funcs as fcs

# def C_91():

# def C_92():

def C_93(α,β,mH,l5,mh=fcs.mh):
    '''Quartic coupling H,H,H,H'''
    return (-3j/(fcs.v**2*np.sin(2*β)**2))*((np.cos(β-α)*np.sin(2*α)-2*np.sin(β+α))**2*mH**2-(np.cos(2*α)-np.cos(2*β))**2*l5*fcs.v**2/2+mh**2*np.sin(2*α)**2*np.sin(β-α)**2)

def C_94(α,β,mH,mA,l5,mh=fcs.mh):
    '''Quartic coupling h,h,A,A'''
    return (1j/(fcs.v**2*np.sin(2*β)**2))*(np.cos(β-α)*np.sin(2*α)*(np.cos(β-α)*np.sin(2*β)-2*np.sin(β+α))*mH**2+l5*fcs.v**2*(np.cos(β+α)**2+np.cos(2*β)**2*np.cos(β-α)**2)-2*mA**2*np.sin(2*β)**2*np.sin(β-α)**2-(2*np.cos(β+α)+np.sin(2*α)*np.sin(β-α))*(2*np.cos(β+α)-np.sin(2*β)*np.sin(β-α))*mh**2)

# def C_97():

# def C_100():

# def C_107():

# def C_111():

# def C_128():    

def C_102(α,β,mH,mA,l5,mh=fcs.mh):
    '''Quartic coupling H,H,G0,G0'''
    return (-1j/(fcs.v**2*np.sin(2*β)))*(np.sin(2*β)*mH**2-np.sin(2*α)*(mh**2-mH**2)*np.sin(β-α)**2+np.sin(2*β)*(2*mA**2-l5*fcs.v**2)*np.sin(β-α)**2)

def C_123(α,β,mH,l5,mh=fcs.mh):
    '''Quartic coupling A,A,A,A'''
    return (-3j/(fcs.v**2*np.sin(2*β)**2))*((2*np.cos(β+α)-np.sin(2*β)*np.sin(β-α))**2*mh**2+(np.cos(β-α)*np.sin(2*β)-2*np.sin(β+α))**2*mH**2-2*l5*fcs.v**2*np.cos(2*β)**2)

def C_140(α,β,mH,l5,mh=fcs.mh):
    '''Quartic coupling H+,H+,H-,H-'''
    return (-2j/(fcs.v**2*np.sin(2*β)**2))*((2*np.cos(β+α)-np.sin(2*β)*np.sin(β-α))**2*mh**2+(np.cos(β-α)*np.sin(2*β)-2*np.sin(β+α))**2*mH**2-2*l5*fcs.v**2*np.cos(2*β)**2)
    