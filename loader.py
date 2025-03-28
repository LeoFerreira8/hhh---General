"""
Created on Sun May 26 00:13:53 2024

@author: leo
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scan_parameterspace_funcs as fcs
import scan_parameterspace as spr
import quartic_couplings as qtcp
import bsg
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

#%%%                                Model parameters to load

small_l5 = True
alignment = True
non_alignment_max = 0.2
load_CDF = False
Calculate_Teo = False
Load_Teo = False
TwoLoop = False

######## loading

latex_model = '2HDM - Type II'
    
if alignment:
    latex_alignment = r'($\beta-\alpha=\frac{\pi}{2}$)'
else:
    latex_alignment = r'($\beta-\alpha\in[\frac{\pi}{2}\pm%.2f]$)' %(non_alignment_max,)

if alignment:
    strga = 'A'
else:
    strg = str(non_alignment_max)
    strga = 'NA'+strg

set_dir = 'data_'+'GTHDMII331lk'+'-'+strga+'/'

TableTot = pd.read_csv('./'+set_dir+'/GTHDMII331lk'+'-'+strga+'-Theo_PDG.csv')
TableTot_STU = pd.read_csv('./'+set_dir+'/GTHDMII331lk'+'-'+strga+'-STU_PDG.csv')
TableTot_STU_Collid = pd.read_csv('./'+set_dir+'/GTHDMII331lk'+'-'+strga+'-Collid_PDG.csv')
TableTot_STU_Collid_BSG = pd.read_csv('./'+set_dir+'/GTHDMII331lk'+'-'+strga+'-BSG_PDG.csv')
TableTot_STU_Collid_BSG_unit = pd.read_csv('./'+set_dir+'/GTHDMII331lk'+'-'+strga+'-PU_PDG.csv')

Dataset_teo = {'name': r'PS', 'data': TableTot}
Dataset_stu = {'name': r'PS+EW', 'data': TableTot_STU}
Dataset_col = {'name': r'PS+EW+C', 'data': TableTot_STU_Collid}
Dataset_bsg = {'name': r'PS+EW+C+bs$\gamma$', 'data': TableTot_STU_Collid_BSG}
Dataset_unit = {'name': r'PS+EW+C+bs$\gamma$+PU', 'data': TableTot_STU_Collid_BSG_unit}

#%%                                 Plots

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
smarker = 1

resol=400
figsiz = 3.5
clbsz = 8

def str_to_tex(strg):
    latex_parameters = [r'$m_A$ [GeV]',r'$m_H$ [GeV]',r'$m_{H^\pm}$ [GeV]',r'$\cos{\alpha}$',r'$\tan{\beta}$',r'$M$ [GeV]',r'$m_{12}^2$ [GeV$^2$]', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$', r'$\lambda_5$',r'$\kappa_\lambda$',r'$\kappa_\lambda^{(0)}$',r'$\tilde{\kappa}_\lambda$',r'$\bar{\kappa}_\lambda$',r'$\sin{(\beta-\alpha)}$',r'$C_{93}$',r'$C_{94}$',r'$C_{102}$',r'$C_{123}$',r'$C_{140}$', r'$\kappa_{\lambda}^{\rm{NNLO}}$', r'$\kappa_{\lambda}^{\rm{NNLO}}/\kappa_\lambda^{\rm{NLO}}$']
    
    if strg=='mA':
        return latex_parameters[0]
    elif strg=='mH':
        return latex_parameters[1]
    elif strg=='mHpm':
        return latex_parameters[2]
    elif strg=='cosa':
        return latex_parameters[3]
    elif strg=='tanb':
        return latex_parameters[4]
    elif strg=='M':
        return latex_parameters[5]
    elif strg=='m122':
        return latex_parameters[6]
    elif strg=='l1':
        return latex_parameters[7]
    elif strg=='l2':
        return latex_parameters[8]
    elif strg=='l3':
        return latex_parameters[9]
    elif strg=='l4':
        return latex_parameters[10]
    elif strg=='l5':
        return latex_parameters[11]
    elif strg=='kappa':
        return latex_parameters[12]
    elif strg=='kappa-tree':
        return latex_parameters[13]
    elif strg=='kappa-kan-x':
        return latex_parameters[14]
    elif strg=='kappa-kan':
        return latex_parameters[15]
    elif strg=='sino':
        return latex_parameters[16]
    elif strg=='c93':
        return latex_parameters[17]
    elif strg=='c94':
        return latex_parameters[18]
    elif strg=='c102':
        return latex_parameters[19]
    elif strg=='c123':
        return latex_parameters[20]
    elif strg=='c140':
        return latex_parameters[21]
    elif strg=='k NNLO':
        return latex_parameters[22]
    elif strg=='k2/k1':
        return latex_parameters[23]
    else:
        raise ValueError("Invalid parameter.")
    

def plotter_3(param1,param2,param3,*dataset,**kwargs):
    
    filename=kwargs.get('filename')

    fig, ax = plt.subplots(figsize=(figsiz, figsiz),dpi=resol)
    
    plt.title(latex_model +' '+ latex_alignment,size=8)
    
    for tbl in dataset:
        sino = np.sin(fcs.beta(tbl['data']['tanb'])-fcs.alpha(tbl['data']['cosa']))
        proxer=tbl['data'].copy()
        proxer.insert(0,"sino", sino)
        proxer = proxer.sort_values(by=[param3],ascending=True)
        plt.scatter(proxer[param1],proxer[param2],c=proxer[param3],s=smarker,rasterized=True)
        plt.legend(title=tbl['name'],title_fontproperties={'weight': 'bold','size': 8})
        
    cbar=plt.colorbar(label=str_to_tex(param3))
    plt.xlabel(str_to_tex(param1))
    #plt.xlim(0.98,1.00)
    plt.ylabel(str_to_tex(param2))
    #plt.ylim(0,40)
    ax.grid()
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', right=True, top=True, direction='in')
    
    cbar.ax.tick_params(labelsize=clbsz)  # Change size of tick labels
    cbar.ax.yaxis.label.set_size(clbsz)   # Change size of colorbar label
    
    if filename:
        fig.savefig("../Figs/"+filename+".pdf",format='pdf',dpi=resol,bbox_inches='tight')
    
    plt.show()

    return 0

def plotter_3_extra(param1,param2,param3,*dataset,**kwargs):

    filename=kwargs.get('filename')

    fig, ax = plt.subplots(figsize=(figsiz, figsiz),dpi=resol)
    
    plt.title(latex_model +' '+ latex_alignment,size=8)
    
    tbl=dataset[0]
    hexbinnin=dataset[1]
    
    sino = np.sin(fcs.beta(tbl['data']['tanb'])-fcs.alpha(tbl['data']['cosa']))
    proxer=tbl['data'].copy()
    proxer.insert(0,"sino", sino)
    proxer = proxer.sort_values(by=[param3],ascending=True)
    plt.scatter(proxer[param1],proxer[param2],c=proxer[param3],s=smarker,rasterized=True)    

    cbar=plt.colorbar(label=str_to_tex(param3))
    
    sino = np.sin(fcs.beta(hexbinnin['data']['tanb'])-fcs.alpha(hexbinnin['data']['cosa']))
    proxer=hexbinnin['data'].copy()
    proxer.insert(0,"sino", sino)
    
    #plt.scatter(proxer[param1], proxer[param2], c='r',alpha=0.4,s=.01)
    sns.kdeplot(x=proxer[param1], y=proxer[param2], levels=[0.015], colors='red', linewidths=1)
    
    handles = [
    Line2D([0], [0], marker='o',color='w', markerfacecolor='k', markersize=5, label=tbl['name']),
    Line2D([0], [0], color='red', lw=1, label=hexbinnin['name'])
    ]
    plt.legend(handles=handles,title_fontproperties={'weight': 'bold','size': 8})
    plt.xlabel(str_to_tex(param1))
    plt.xlim(0.98,1.00)
    plt.ylabel(str_to_tex(param2))
    #plt.ylim(0.8,40)
    ax.grid()
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', right=True, top=True, direction='in')
    cbar.ax.tick_params(labelsize=clbsz)  # Change size of tick labels
    cbar.ax.yaxis.label.set_size(clbsz)   # Change size of colorbar label
    
    if filename:
        fig.savefig("../Figs/"+filename+".pdf",format='pdf',dpi=resol,bbox_inches='tight')
    
    plt.show()

    return 0

def plotter_comp(param1,param2,param3,param4,*dataset):    

    fig, ax = plt.subplots(figsize=(10, 10))
    
    plt.title(latex_model +' '+ latex_alignment,size=20)
    
    for tbl in dataset:
        ratio = tbl['data'][param3]/tbl['data'][param4]
        if param1=='sino':
            sino = np.sin(fcs.beta(tbl['data']['tanb'])-fcs.alpha(tbl['data']['cosa']))
            plt.scatter(sino,tbl['data'][param2],c=ratio,s=smarker,rasterized=True)
        elif param2=='sino':
            sino = np.sin(fcs.beta(tbl['data']['tanb'])-fcs.alpha(tbl['data']['cosa']))
            plt.scatter(tbl['data'][param1],sino,c=ratio,s=smarker,rasterized=True)
        else:
            plt.scatter(tbl['data'][param1],tbl['data'][param2],c=ratio,s=smarker,rasterized=True)
    cbar=plt.colorbar(label=str_to_tex(param3)+'/'+str_to_tex(param4))
    plt.xlabel(str_to_tex(param1), size=25)
    plt.xticks(size=20)
    #plt.xlim(125,900)
    plt.ylabel(str_to_tex(param2), size=25)
    plt.yticks(size=20)
    #plt.ylim(125,900)
    ax.grid()
    cbar.ax.tick_params(labelsize=25)  # Change size of tick labels
    cbar.ax.yaxis.label.set_size(25)   # Change size of colorbar label
    
    plt.show()

    return 0

def plotter_diff(param1,param2,param3,param4,*dataset,alph=1,**kwargs):
    
    filename=kwargs.get('filename')
    
    #colors = ['firebrick','royalblue','khaki','aquamarine','yellowgreen']

    fig, ax = plt.subplots(figsize=(figsiz, figsiz),dpi=resol)
    
    plt.title(latex_model +' '+ latex_alignment,size=8)
    
    for i,tbl in enumerate(dataset):
        plt.scatter(tbl['data'][param1]-tbl['data'][param2],tbl['data'][param3]-tbl['data'][param4],label=tbl['name'],alpha=alph**i,s=smarker,rasterized=True)
        
    plt.xlabel(str_to_tex(param1).replace(' [GeV]','')+'-'+str_to_tex(param2))
    #plt.ylim(-75,720)
    plt.ylabel(str_to_tex(param3).replace(' [GeV]','')+'-'+str_to_tex(param4))
    #plt.xlim(-270,520)
    ax.grid()
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', right=True, top=True, direction='in')
    ax.set_xticks([-250,0,250,500])
    ax.set_yticks([0,250,500,750])
    
    plt.legend(loc='upper left')
    
    if filename:
        fig.savefig("../Figs/"+filename+".pdf",format='pdf',dpi=resol,bbox_inches='tight')
    
    plt.show()

    return 0

def plotter_diff_contour(param1,param2,param3,param4,*dataset,**kwargs):
    
    colors = ['b','orange','g','r']
    
    filename = kwargs.get('filename')

    fig, ax = plt.subplots(figsize=(figsiz, figsiz),dpi=resol)
    
    plt.title(latex_model +' '+ latex_alignment,size=8)
    
    handles = []
    for i,tbl in enumerate(dataset):
        #plt.scatter(tbl['data'][param1]-tbl['data'][param2],tbl['data'][param3]-tbl['data'][param4],label=tbl['name'],alpha=0.1)
        sns.kdeplot(x=tbl['data'][param1]-tbl['data'][param2], y=tbl['data'][param3]-tbl['data'][param4], levels=[0.015], colors=colors[i], linewidths=1)
        
        handles.append(
        Line2D([0], [0], color=colors[i], lw=1, label=tbl['name'])
        )
    plt.xlabel(str_to_tex(param1).replace(' [GeV]','')+'-'+str_to_tex(param2))
    #plt.ylim(-75,720)
    plt.ylabel(str_to_tex(param3).replace(' [GeV]','')+'-'+str_to_tex(param4))
    #plt.xlim(-270,520)
    ax.grid()
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', right=True, top=True, direction='in')
    
    ax.set_xticks([-250,0,250,500])
    ax.set_yticks([0,250,500,750])
    
    plt.legend(handles=handles,loc='upper left')
    
    if filename:
        fig.savefig("../Figs/"+filename+".pdf",format='pdf',dpi=resol,bbox_inches='tight')
    
    plt.show()

    return 0

def plotter_diff_color(param1,param2,param3,param4,param5,*dataset,**kwargs):
    
    filename=kwargs.get('filename')

    fig, ax = plt.subplots(figsize=(figsiz, figsiz),dpi=resol)
    
    plt.title(latex_model +' '+ latex_alignment,size=8)
    
    for tbl in dataset:
        sino = np.sin(fcs.beta(tbl['data']['tanb'])-fcs.alpha(tbl['data']['cosa']))
        proxer=tbl['data'].copy()
        proxer.insert(0,"sino", sino)
        proxer = proxer.sort_values(by=[param5],ascending=True)
        plt.scatter(proxer[param1]-proxer[param2],proxer[param3]-proxer[param4],c=proxer[param5],s=smarker,rasterized=True)
        plt.legend(title=tbl['name'],title_fontproperties={'weight': 'bold','size': 8},loc='upper left')
        
    cbar=plt.colorbar(label=str_to_tex(param5))
    #cbar.ax.axhline(2.6, c='r')
    plt.xlabel(str_to_tex(param1).replace(' [GeV]','')+'-'+str_to_tex(param2))
    #plt.xlim(-300,520)
    plt.ylabel(str_to_tex(param3).replace(' [GeV]','')+'-'+str_to_tex(param4))
    #plt.ylim(-80,700)
    ax.grid()
    cbar.ax.tick_params(labelsize=clbsz)  # Change size of tick labels
    cbar.ax.yaxis.label.set_size(clbsz)   # Change size of colorbar label
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', right=True, top=True, direction='in')
    #ax.set_xticks([-250,0,250,500])
    #ax.set_yticks([0,250,500,750])
    #ax.set_xticks([-50,0,50,100,150,200,250])
    #ax.set_yticks([0,50,100,150,200,250])
    #ax.set_xticks([-100,-50,0,50,100,150])
    #ax.set_yticks([-50,0,-150,-100,-50,0,50,100])
    
    if filename:
        fig.savefig("../Figs/"+filename+".pdf",format='pdf',dpi=resol,bbox_inches='tight')
    
    plt.show()

    return 0

def plotter_diff_comp(param1,param2,param3,param4,param5,param6,*dataset):    

    fig, ax = plt.subplots(figsize=(10, 10),dpi=resol)
    
    plt.title(latex_model +' '+ latex_alignment,size=8)
    
    for tbl in dataset:
        ratio = tbl['data'][param5]/tbl['data'][param6]
        proxer=tbl['data'].copy()
        proxer.insert(0,"ratio", ratio)
        proxer = proxer.sort_values(by=['ratio'],ascending=True)
        plt.scatter(proxer[param1]-proxer[param2],proxer[param3]-proxer[param4],c=proxer['ratio'],s=smarker,rasterized=True)
    cbar=plt.colorbar(label=str_to_tex(param5)+'/'+str_to_tex(param6))
    plt.xlabel(str_to_tex(param1).replace(' [GeV]','')+'-'+str_to_tex(param2), size=25)
    plt.xticks(size=20)
    plt.xlim(-450,200)
    plt.ylabel(str_to_tex(param3).replace(' [GeV]','')+'-'+str_to_tex(param4), size=25)
    plt.yticks(size=20)
    plt.ylim(-320,350)
    ax.grid()
    cbar.ax.tick_params(labelsize=20)  # Change size of tick labels
    cbar.ax.yaxis.label.set_size(20)   # Change size of colorbar label
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', right=True, top=True, direction='in')
    
    plt.show()

    return 0

#%%%  Examples

# plotter_diff_color('mH','mHpm','mA','mHpm','kappa',Dataset_bsg)
# plotter_diff_color('mH','mHpm','mA','mHpm','tanb',Dataset_bsg)
# plotter_diff('mH','mHpm','mA','mHpm',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg)
# plotter_diff_comp('mH','mHpm','mA','mHpm','kappa','kappa-kan',Dataset_bsg)
# plotter('mH','mHpm',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg)
# plotter('mA','mHpm',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg)
# plotter('mHpm','tanb',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg)
# plotter('mHpm','M',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg)
# plotter_3('mHpm','M','kappa',Dataset_bsg)
# plotter('M','tanb',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg)
# plotter('M','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg)
# plotter('tanb','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg)
# plotter('mA','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg)
# plotter('mH','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg)
# plotter('mHpm','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg)
# plotter('l1','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg)
# plotter('l2','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg)
# plotter('l3','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg)
# plotter('l4','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg)
# plotter('l5','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg)

# plotter_diff_color('mH','mHpm','mA','mHpm','kappa',Dataset_unit)
# plotter_diff_color('mH','mHpm','mA','mHpm','tanb',Dataset_unit)
# plotter_diff('mH','mHpm','mA','mHpm',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg,Dataset_unit)
# plotter_diff_comp('mH','mHpm','mA','mHpm','kappa','kappa-kan',Dataset_unit)
# plotter('mH','mHpm',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg,Dataset_unit)
# plotter('mA','mHpm',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg,Dataset_unit)
# plotter('mHpm','tanb',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg,Dataset_unit)
# plotter('mHpm','M',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg,Dataset_unit)
# plotter('M','tanb',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg,Dataset_unit)
# plotter('M','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg,Dataset_unit)
# plotter('tanb','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg,Dataset_unit)
# plotter('mA','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg,Dataset_unit)
# plotter('mH','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg,Dataset_unit)
# plotter('mHpm','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg,Dataset_unit)
# plotter('l1','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg,Dataset_unit)
# plotter('l2','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg,Dataset_unit)
# plotter('l3','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg,Dataset_unit)
# plotter('l4','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg,Dataset_unit)
# plotter('l5','kappa',Dataset_teo,Dataset_stu,Dataset_col,Dataset_bsg,Dataset_unit)
# plotter('c93','kappa',Dataset_unit)
# plotter('c94','kappa',Dataset_unit)
# plotter('c102','kappa',Dataset_unit)
# plotter('c123','kappa',Dataset_unit)
# plotter('c140','kappa',Dataset_unit)

#%%

plotter_diff_color('mHpm','M','mH','M','kappa',Dataset_unit,filename='Kappa_final')
plotter_diff_color('mHpm','M','mA','M','kappa',Dataset_unit)
plotter_diff_color('mHpm','M','mH','M','tanb',Dataset_unit,filename='tanb_final')
plotter_diff_color('mHpm','M','mA','M','tanb',Dataset_unit)


#%%                             Obtaining the constraints parameter space

if Calculate_Teo:
    TableTot_unit = pd.concat([TableTot,pd.DataFrame(np.array(spr.calculate_eigenvalues(TableTot)).T,columns=['a0'])],axis=1)
    
    cnd = spr.perturbative_unitarity_const_a0(TableTot_unit['a0'])
    TableTot_true_teo = TableTot_unit.drop(TableTot_unit[cnd].index)
    
    TableTot_true_teo.to_csv('./'+set_dir+'/THDM'+strgl5+'-'+strga+'_true_teo.csv',index=False)

#%%

if Load_Teo:
    TableTot_true_teo = pd.read_csv('./'+set_dir+'/THDM'+strgl5+'-'+strga+'_true_teo.csv')
    
    Dataset_true_teo = {'name': r'T', 'data': TableTot_true_teo}

#%%                                 Theoretical

plotter_diff_color('mHpm','M','mH','M','kappa',Dataset_teo,filename='Kappa_mHpm_x_mH_M_331lk_PS')
if Load_Teo:
    plotter_diff_color('mHpm','M','mH','M','kappa',Dataset_true_teo,filename='Kappa_mHpm_x_mH_M_331lk_teo')

#%%                                 STU

Dataset_stu_only = {'name': r'EW', 'data': TableTot_STU}

plotter_diff_color('mHpm','M','mH','M','kappa',Dataset_stu_only,filename='Kappa_mHpm_x_mH_M_331lk_stu')


#%%                                 Collider

#%                                  HiggsBounds

cnd = spr.collider_const(TableTot['HiggsB'])
TableTot_Collid = TableTot.drop(TableTot[cnd].index)

#%                                 Impose bounds from HiggsSignals

cnd = spr.signals_const(np.array(TableTot_Collid['HiggsS'],dtype=float))
TableTot_Collid = TableTot_Collid.drop(TableTot_Collid[cnd].index)

Dataset_collid_only = {'name': r'C', 'data': TableTot_Collid}

#%%

plotter_diff_color('mHpm','M','mH','M','kappa',Dataset_collid_only,filename='Kappa_mHpm_x_mH_M_331lk_coll')

#%%                                 BSG

cnd = bsg.Constraints_BSG(np.array(TableTot['tanb'],dtype=float), np.array(TableTot['mHpm'],dtype=float))
TableTot_BSG = TableTot.drop(TableTot[cnd].index)

Dataset_bsg_only = {'name': r'bs$\gamma$', 'data': TableTot_BSG}

plotter_diff_color('mHpm','M','mH','M','kappa',Dataset_bsg_only,filename='Kappa_mHpm_x_mH_M_331lk_bsg')

#%%         

if Load_Teo:
    plotter_diff('mHpm','M','mH','M',Dataset_true_teo,alph=0.8)
    plotter_diff('mHpm','M','mH','M',Dataset_true_teo,Dataset_collid_only,alph=0.8)
    plotter_diff('mHpm','M','mH','M',Dataset_true_teo,Dataset_collid_only,Dataset_bsg_only,alph=0.8)
    plotter_diff('mHpm','M','mH','M',Dataset_true_teo,Dataset_collid_only,Dataset_bsg_only,Dataset_stu_only,filename='Constraints_comp')

    plotter_diff_contour('mHpm','M','mH','M',Dataset_true_teo,Dataset_collid_only,Dataset_bsg_only,Dataset_stu_only,filename='Constraints_comp_1')

    plotter_diff('mHpm','M','mH','M',Dataset_teo,Dataset_true_teo,filename='Teo_Constraints_comp')

#%%                                 Sin x tanb

plotter_3('sino', 'tanb', 'kappa', Dataset_col,filename='Col_everything')
plotter_3('sino', 'tanb', 'kappa', Dataset_collid_only)
plotter_3('sino', 'tanb', 'kappa', Dataset_teo)
#%%

plotter_3_extra('sino', 'tanb', 'kappa', Dataset_teo,Dataset_collid_only,filename='Theoretical_Col_comp')
