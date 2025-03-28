#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 22:23:32 2025

@author: leo
"""

from anyBSM import anyBSM
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

plt.rc('text', usetex=True)
plt.rcParams.update({
    'xtick.major.size': 2.5,       # Major tick marker size on x-axis
    'ytick.major.size': 2.5,       # Major tick marker size on y-axis
    'xtick.major.width': 0.8,       # Major tick marker size on x-axis
    'ytick.major.width': 0.8,       # Major tick marker size on y-axis
    'xtick.minor.size': 1.25,       # Major tick marker size on x-axis
    'ytick.minor.size': 1.25,       # Major tick marker size on y-axis
    'xtick.minor.width': 0.65,       # Major tick marker size on x-axis
    'ytick.minor.width': 0.65,       # Major tick marker size on y-axis
    'xtick.labelsize': 9,        # Font size for x-axis tick labels
    'ytick.labelsize': 9,        # Font size for y-axis tick labels
    'axes.labelsize': 9,         # Font size for x and y axis labels
    'legend.fontsize': 7,        # Font size for the legend
    'font.size': 7               # Global font size (text, title, etc.)
})
mpl.rc('lines', linewidth=0.8)

resol=400

THDM2 = anyBSM('GTHDMII', scheme_name = 'OSalignment')

scale = np.geomspace(1e-7, 1e1,100)

mA = 706.8
mH = 926
mHpm = 935.7
tb = 1.0363
sb = np.sin(np.arctan(tb))
cb = np.cos(np.arctan(tb))
sBmA = 1

mh = 125.1

MW = 80.379
MZ = 91.187
alphaQED = 137.035999679
e = 2*np.sqrt(np.pi/(alphaQED))

v = 2*MW*np.sqrt(1-MW**2/MZ**2)/e
lambdahhhSM = 3*mh**2/v

δΓ = np.copy(scale)
δΓ0 = np.copy(scale)

for i, ge in enumerate(scale):
    
    Ms = np.sqrt(mA**2+(ge+(1/2)*ge*cb/sb+(1/2)*ge*sb/cb)*(v**2))
    
    WmassStd = 'MWm'
    THDM2.setparameters({'Mh2': mH, 'MAh2': mA, 'MHm2': mHpm, 'TanBeta': tb, 'SinBmA': sBmA,'M': Ms, 'rLam6': ge, 'rLam7': ge}) #Define new mass in anyBSM
    THDM2.progress=False
    THDM2.warnSSSS=False
    dic = THDM2.lambdahhh()
    δΓ[i] = np.real(dic['total'])/lambdahhhSM  #Recalculate lambda
    δΓ0[i] = np.real(dic['treelevel'])/lambdahhhSM  #Recalculate lambda
    
    
#%%%

fig, ax = plt.subplots(figsize=(3.5, 3.5),dpi=resol)

plt.plot(scale, δΓ,label='One-loop')
#plt.plot(scale, δΓ0,label='Tree')
plt.ylabel(r'$\kappa_\lambda$')
plt.xlabel(r'$\Lambda_x$')
ax.set_xscale('log')
plt.ylim(-20,50)
plt.minorticks_on()
plt.tick_params(axis='both', which='both', right=True, top=True, direction='in')
ax.grid()

textstr = '\n'.join((
    r'\textbf{2HDM-II}',
    r'$m_A=%d$ GeV' %(mA),
    r'$m_{H}=%d$ GeV' %(mH),
    r'$m_{H^\pm}=%d$ GeV' %(mHpm),
    r'$\sin(\beta-\alpha)=%.2f$' %(sBmA),
    r'$\tan(\beta)=%.1f$' %(tb),
    ))

ax.text(0.06, 0.72, textstr, transform=ax.transAxes, verticalalignment='top')

plt.legend(loc='upper left')

plt.gcf()

#fig.set_size_inches(3,3)

fig.savefig("../Figs/scale.pdf",format='pdf',dpi=resol,bbox_inches='tight')

plt.show()