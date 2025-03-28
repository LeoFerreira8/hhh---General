#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:37:49 2024

@author: leo
"""

import numpy as np
from scipy.special import spence
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.ticker import ScalarFormatter, NullFormatter, AutoMinorLocator
import scan_parameterspace_funcs as fcs

#implementing CKM

theta12 = np.arcsin(0.22650)
theta23 = np.arcsin(0.04053)
theta13 = np.arcsin(0.00361)
delta = 1.196

c12 = np.cos(theta12)
s12 = np.sin(theta12)
c23 = np.cos(theta23)
s23 = np.sin(theta23)
c13 = np.cos(theta13)
s13 = np.sin(theta13)
cos = np.cos
sin = np.sin

V = np.matmul([[1,0,0],[0,c23,s23],[0,-s23,c23]],np.matmul([[c13,0,s13*np.exp(-1j*delta)],[0,1,0],[-s13*np.exp(1j*delta),0,c13]],[[c12,s12,0],[-s12,c12,0],[0,0,1]]))

#implementing Vu and mass matrices

mu = 2e-3
md = 3e-3
mc = 90
ms = 70e-3
mt = 160
mb = 4

#Matrix definitions

def Vu(φ,θ,ψ):
    return np.matmul([[cos(φ),-sin(φ),0],[sin(φ),cos(φ),0],[0,0,1]],np.matmul([[cos(θ),0,-sin(θ)],[0,1,0],[sin(θ),0,cos(θ)]],[[1,0,0],[0,cos(ψ),-sin(ψ)],[0,sin(ψ),cos(ψ)]]))
    
def Γau(φ,θ,ψ,tb):
    z = np.zeros(np.shape(tb))
    A = np.stack([-tb,z,z,z,-tb,z,z,z,1/tb], axis=-1).reshape(*np.shape(tb), 3, 3)
    #B = np.stack([B00, B01, B10, B11], axis=-1).reshape(-1, 2, 2)
    return np.matmul(Vu(φ, θ, ψ).T,np.matmul(A,Vu(φ, θ, ψ)))

def Γad(φ,θ,ψ,tb):
    z = np.zeros(np.shape(tb))
    A = np.stack([1/tb,z,z,z,1/tb,z,z,z,-tb], axis=-1).reshape(*np.shape(tb), 3, 3)
    return np.matmul(np.conjugate(V.T),np.matmul(Vu(φ, θ, ψ).T,np.matmul(A,np.matmul(Vu(φ, θ, ψ),V))))

def mL(φ,θ,ψ,tb):
    return -np.matmul(np.diag([md,ms,mb]),np.matmul(Γad(φ,θ,ψ,tb),np.conj(V.T)))

def mR(φ,θ,ψ,tb):
    return np.matmul(np.conj(V.T),np.matmul(Γau(φ,θ,ψ,tb),np.diag([mu,mc,mt])))

# Auxiliary parameters

def y2(φ, θ, ψ, tb):
    y2 = np.conjugate(mR(φ, θ, ψ, tb))[...,2,2]*mR(φ, θ, ψ, tb)[...,1,2]/(mt**2*V[2,2]*np.conjugate(V[2,1]))
    return y2
    
def xy(φ, θ, ψ, tb):
    xy = np.conjugate(mL(φ, θ, ψ, tb))[...,2,2]*mR(φ, θ, ψ, tb)[...,1,2]/(mt*mb*V[2,2]*np.conjugate(V[2,1]))
    return xy
    
# Auxiliary functions

def G17(x):
    return x*(7-5*x-8*x**2)/(24*(x-1)**3)+x**2*(3*x-2)*np.log(x)/(4*(x-1)**4)
    
def G18(x):
    return x*(2+5*x-x**2)/(8*(x-1)**3)-3*x**2*np.log(x)/(4*(x-1)**4)

def G27(x):
    return x*(3-5*x)/(12*(x-1)**2)+x*(3*x-2)*np.log(x)/(6*(x-1)**3)
    
def G28(x):
    return x*(3-x)/(4*(x-1)**2)-x*np.log(x)/(2*(x-1)**3)

def C17(y):
    term1 = (y * (18 - 37 * y + 8 * y**2)) / (y - 1)**4 * spence(1/y)
    term2 = (y * (-14 + 23 * y + 3 * y**2)) / (y - 1)**5 * np.log(y)**2
    term3 = (-50 + 251 * y - 174 * y**2 - 192 * y**3 + 21 * y**4) / (9 * (y - 1)**5) * np.log(y)
    term4 = -(3 * y - 2) / (3 * (y - 1)**4) * np.log(y)
    term5 = (797 - 5436 * y + 7569 * y**2 - 1202 * y**3) / (108 * (y - 1)**4)
    term6 = -(16 - 29 * y + 7 * y**2) / (18 * (y - 1)**3)
    return 2/9 * y * (term1 + term2 + term3 + term4 + term5 + term6)

def C27(y):
    term1 = 4 * (-3 + 7 * y - 2 * y**2) / (3 * (y - 1)**3) * spence(1/y)
    term2 = (8 - 14 * y - 3 * y**2) / (3 * (y - 1)**4) * np.log(y)**2
    term3 = 2 * (-3 - y + 12 * y**2 - 2 * y**3) / (3 * (y - 1)**4) * np.log(y)
    term4 = (7 - 13 * y + 2 * y**2) / (y - 1)**3
    return -(4/3) * y * (term1 + term2 + term3 + term4)

def C18(y):
    term1 = (y * (30 - 17 * y + 13 * y**2)) / (y - 1)**4 * spence(1/y)
    term2 = -(y * (31 + 17 * y)) / (y - 1)**5 * np.log(y)**2
    term3 = -(226 - 817 * y - 1353 * y**2 - 318 * y**3 - 42 * y**4) / (36 * (y - 1)**5) * np.log(y)
    term4 = -(3 * y - 2) / (6 * (y - 1)**4) * np.log(y)
    term5 = (1130 - 18153 * y + 7650 * y**2 - 4451 * y**3) / (216 * (y - 1)**4)
    term6 = -(16 - 29 * y + 7 * y**2) / (36 * (y - 1)**3)
    return 1/6 * y * (term1 + term2 + term3 + term4 + term5 + term6)

def C28(y):
    term1 = -36 + 25 * y - 17 * y**2 / (2 * (y - 1)**3) * spence(1/y)
    term2 = (19 + 17 * y) / (3 * (y - 1)**4) * np.log(y)**2
    term3 = (-3 - 187 * y + 12 * y**2 - 14 * y**3) / (4 * (y - 1)**4) * np.log(y)
    term4 = 3 * (143 - 44 * y + 29 * y**2) / (8 * (y - 1)**3)
    return -(1/3) * y * (term1 + term2 + term3 + term4)

def D17(y):
    term1 = (-31 - 18 * y + 135 * y**2 - 14 * y**3) / (6 * (y - 1)**4)
    term2 = (y * (14 - 23 * y - 3 * y**2)) / (y - 1)**5 * np.log(y)
    return 2/9 * y * (term1 + term2)

def D27(y):
    term1 = (21 - 47 * y + 8 * y**2) / (y - 1)**3
    term2 = 2 * (-8 + 14 * y + 3 * y**2) / (y - 1)**4 * np.log(y)
    return -(2/9) * y * (term1 + term2)

def D18(y):
    term1 = (-38 - 261 * y + 18 * y**2 - 7 * y**3) / (6 * (y - 1)**4)
    term2 = (y * (31 + 17 * y)) / (y - 1)**5 * np.log(y)
    return 1/6 * y * (term1 + term2)

def D28(y):
    term1 = (81 - 16 * y + 7 * y**2) / (2 * (y - 1)**3)
    term2 = -(19 + 17 * y) / (y - 1)**4 * np.log(y)
    return -(1/3) * y * (term1 + term2)

def c7(Var, mhp):
    term1 = (Var[0]**2 / 3) * G17(mt**2 / mhp**2)
    term2 = Var[1] * G27(mt**2 / mhp**2)
    return term1 + term2

def c8(Var, mhp):
    term1 = (Var[0]**2 / 3) * G18(mt**2 / mhp**2)
    term2 = Var[1] * G28(mt**2 / mhp**2)
    return term1 + term2

def cn7(Var, mhp):
    term1 = Var[0]**2 * C17(mt**2 / mhp**2)
    term2 = Var[1] * C27(mt**2 / mhp**2)
    term3 = (Var[0]**2 * D17(mt**2 / mhp**2) + Var[1] * D27(mt**2 / mhp**2)) * np.log(160**2 / mhp**2)
    return term1 + term2 + term3

def cn8(Var, mhp):
    term1 = Var[0]**2 * C18(mt**2 / mhp**2)
    term2 = Var[1] * C28(mt**2 / mhp**2)
    term3 = (Var[0]**2 * D18(mt**2 / mhp**2) + Var[1] * D28(mt**2 / mhp**2)) * np.log(160**2 / mhp**2)
    return term1 + term2 + term3

def bsg(Var, mhp):
    c7_val = c7(Var, mhp)
    c8_val = c8(Var, mhp)
    cn7_val = cn7(Var, mhp)
    cn8_val = cn8(Var, mhp)
    
    rv = np.abs(V[2,1]*V[2,2])**2/np.abs(V[1,2])**2

    delta_value = 1e-4 * (rv / 0.9626) * np.real(
        -8.100 * c7_val - 2.509 * c8_val + 2.767 * c7_val * np.conjugate(c8_val) +
        5.348 * np.abs(c7_val)**2 + 0.89 * np.abs(c8_val)**2 - 0.085 * cn7_val - 0.025 * cn8_val +
        0.095 * c7_val * np.conjugate(cn7_val) + 0.008 * c8_val * np.conjugate(cn8_val) +
        0.028 * (c7_val * np.conjugate(cn8_val) + cn7_val * np.conjugate(c8_val))
    )

    bsm = 3.36*(rv/0.9626)*1e-4
    
    return bsm + delta_value

def Constraints_BSG(tb,mhp, φ=0, θ=0, ψ=0):
    exp = [0.000302,0.000362]
    
    if fcs.THDM_type == "":
        Var = [1/tb,1/tb**2]
    elif fcs.THDM_type == "II":
       if fcs.small_l5:
           Var = [y2(φ, θ, ψ, tb),xy(φ, θ, ψ, tb)]
       else:
           Var = [1/tb,-1+tb*0]
    
    return np.where((exp[0]<bsg(Var, mhp))
                    & (bsg(Var, mhp)<exp[1])
                    ,False,True)

# Model = "II"

# if Model == "":
#     Var = lambda tb: [1/tb,1/tb**2]
# elif Model == "II":
#     Var = lambda tb: [1/tb,-1+tb*0]
# else:
#     Var = lambda tb, φ, θ, ψ: [y2(φ, θ, ψ, tb),xy(φ, θ, ψ, tb)]
        

# fig, ax = plt.subplots(figsize=(10, 10))

# tb_A = np.linspace(0.8, 50,1000)
# mhp_A = np.linspace(200, 1000,1000)

# Xi, Yi = np.meshgrid(mhp_A, tb_A)
# zi = bsg(Var(Yi), Xi)

# #plt.title(latex_model +' '+ latex_alignment,size=20)

# levels = [0.000302,0.000362]
# #for tbl in dataset:
# #   ratio = tbl[param5]/tbl[param6]
# ax.contourf(Xi, Yi,zi,levels=levels)
# plt.xlabel(r'$M_{H^{\pm}}$ [GeV]', size=25)
# plt.xticks(size=30)
# ax.xaxis.set_minor_locator(AutoMinorLocator())
# ax.tick_params(top=True,which='both')
# ax.tick_params(right=True, which='both')
# #plt.xlim(-450,200)
# plt.ylabel(r'$\tan{\beta}$', size=25)
# plt.yscale("log")
# ax.yaxis.set_major_formatter(ScalarFormatter())
# ax.yaxis.set_minor_formatter(NullFormatter())
# ax.set_yticks([1,5,10,50])
# plt.yticks(size=30)
# plt.ylim(0.8,50)
# ax.grid()
# #cbar.ax.tick_params(labelsize=20)  # Change size of tick labels
# #cbar.ax.yaxis.label.set_size(20)   # Change size of colorbar label

# plt.show()

# #%%

# Model = "331"

# if Model == "":
#     Var = lambda tb: [1/tb,1/tb**2]
# elif Model == "II":
#     Var = lambda tb: [1/tb,-1+tb*0]
# else:
#     Var = lambda tb, φ, θ, ψ: [np.sqrt(y2(φ, θ, ψ, tb)),-xy(φ, θ, ψ, tb)]

# fig, ax = plt.subplots(figsize=(10, 10))

# tb_A = np.linspace(0.8, 50,1000)
# mhp_A = np.linspace(200, 1000,1000)

# Xi, Yi = np.meshgrid(mhp_A, tb_A)
# # zi = np.copy(Yi)
# # for i,p in enumerate(tb_A):
# #     for j,l in enumerate(mhp_A):
# #         zi[i,j] = bsg(Var(p,0,0,0), l)
# zi = bsg(Var(Yi,0,0,0), Xi)

# #plt.title(latex_model +' '+ latex_alignment,size=20)

# levels = [0.000302,0.000362]
# #for tbl in dataset:
# #   ratio = tbl[param5]/tbl[param6]
# ax.contourf(Xi, Yi,zi,levels=levels)
# plt.xlabel(r'$M_{H^{\pm}}$ [GeV]', size=25)
# plt.xticks(size=30)
# ax.xaxis.set_minor_locator(AutoMinorLocator())
# ax.tick_params(top=True,which='both')
# ax.tick_params(right=True, which='both')
# #plt.xlim(-450,200)
# plt.ylabel(r'$\tan{\beta}$', size=25)
# plt.yscale("log")
# ax.yaxis.set_major_formatter(ScalarFormatter())
# ax.yaxis.set_minor_formatter(NullFormatter())
# ax.set_yticks([1,5,10,50])
# plt.yticks(size=30)
# plt.ylim(0.8,50)
# ax.grid()
# #cbar.ax.tick_params(labelsize=20)  # Change size of tick labels
# #cbar.ax.yaxis.label.set_size(20)   # Change size of colorbar label

# plt.show()

# #%%

# Model = "331"

# if Model == "":
#     Var = lambda tb: [1/tb,1/tb**2]
# elif Model == "II":
#     Var = lambda tb: [1/tb,-1+tb*0]
# else:
#     Var = lambda tb, φ, θ, ψ: [np.sqrt(y2(φ, θ, ψ, tb)),-xy(φ, θ, ψ, tb)]

# fig, ax = plt.subplots(figsize=(10, 10))

# psi_A = np.linspace(-np.pi/2, np.pi/2,101)
# theta_A = np.linspace(-np.pi/2, np.pi/2,100)

# Xi, Yi = np.meshgrid(psi_A, theta_A)
# zi = np.copy(Yi)
# for i,p in enumerate(psi_A):
#     for j,l in enumerate(theta_A):
#         zi[j,i] = bsg(Var(50,0,l,p), 300)

# #plt.title(latex_model +' '+ latex_alignment,size=20)

# levels = [0.000302,0.000362]
# #for tbl in dataset:
# #   ratio = tbl[param5]/tbl[param6]
# ax.contourf(Xi, Yi,zi,levels=levels)
# plt.xlabel(r'$\psi$', size=25)
# plt.xticks(size=30)
# ax.xaxis.set_minor_locator(AutoMinorLocator())
# ax.tick_params(top=True,which='both')
# ax.tick_params(right=True, which='both')
# #plt.xlim(-450,200)
# plt.ylabel(r'$\theta$', size=25)
# #plt.yscale("log")
# #ax.yaxis.set_major_formatter(ScalarFormatter())
# #ax.yaxis.set_minor_formatter(NullFormatter())
# #ax.set_yticks([1,5,10,50])
# plt.yticks(size=30)
# #plt.ylim(0.8,50)
# ax.grid()
# #cbar.ax.tick_params(labelsize=20)  # Change size of tick labels
# #cbar.ax.yaxis.label.set_size(20)   # Change size of colorbar label

# plt.show()