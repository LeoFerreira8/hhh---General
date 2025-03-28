"""
Created on Mon Apr 22 11:09:52 2024

@author: leonardoferreira
"""

from anyBSM import anyBSM
import numpy as np
import scan_parameterspace_funcs as fcs
import pandas as pd
import gc
import scan_SPheno_funcs as SPfcs
import scan_higgs_tools_funcs as hggfcs
import bsg
import quartic_couplings as qtcp
from scipy import stats
from time import time
from pathlib import Path

def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'\r Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)

s_mean = -0.05
δs = 0.07

t_mean = 0.00
δt = 0.06

corr = 0.93

N_parameters = 6-1*fcs.alignment
THDM_type = 'THDMIInoZ2'
latex_model = 'General 2HDM'
    
if fcs.alignment:
    latex_alignment = r'($\beta-\alpha=\frac{\pi}{2}$)'
else:
    latex_alignment = r'($\beta-\alpha\in[\frac{\pi}{2}\pm%.2f]$)' %(fcs.non_alignment_max,)

def perturbativity_bounds(l):
    '''Returns True if coupling l is above the perturbativity bound, and False otherwise.'''
    lmax = np.sqrt(4*np.pi)
    lmax = 4*np.pi
    res = np.where(np.abs(l)<lmax,False,True)
    
    return res

def stability_bounds(l_a):
    '''Returns True if array of couplings l_a does not comply with the stability bounds, and False otherwise.'''
    res = np.where(
        (l_a[0] > 0) 
        & (l_a[1] > 0) 
        & (-np.sqrt(l_a[0]*l_a[1]) < l_a[2]) 
        & (l_a[2]+l_a[3]-np.abs(l_a[4]) > -np.sqrt(l_a[0]*l_a[1]))
        ,False,True)
    
    return res

def Spheno_Higgstools_calc_(SPheno_input):
    outpt=pd.DataFrame()
    prog = 1
    tot = SPheno_input.shape[0]
    for index, row in SPheno_input.iterrows():
        progress_bar(prog, tot)
        SPfcs.write_spheno_LHA(list(row))
        success = SPfcs.execute_spheno()
        if success:
            hggs_SPhen_outpt = pd.concat([SPfcs.read_spheno_obs(),hggfcs.Higgs_tools_scan()],axis=1)
            outpt = pd.concat([outpt,hggs_SPhen_outpt])
        else:
            print('SPheno could not calculate %d' %index)
            proxrow = np.copy(row)
            proxrow[5]=-proxrow[5]
            SPfcs.write_spheno_LHA(list(proxrow))
            success = SPfcs.execute_spheno()
            if success:
                outpt = pd.concat([outpt,SPfcs.read_spheno_obs()])
                print('Calculation of %d successful.' %index)
            else:
                print('Failed')

        prog+=1                
                
    outpt = outpt.drop_duplicates()
    
    return outpt

def STU_constraint(s,t,u):
    '''Returns True if STU complies with the bounds, and False otherwise.
    ΔS = 0.00+-0.07 
    ΔT = 0.05+-0.06 
    
    ΔS = -0.01+-0.07 
    ΔT = 0.04+-0.06 
    '''
    
    if fcs.alignment:
        strga = 'A'
    else:
        strg = str(fcs.non_alignment_max)
        strga = 'NA'+strg
        
    if fcs.small_l5:
        strgl5 = '331lk'
    else:
        strgl5 = ''
    
    c12 = corr*δs*δt
    
    #C = np.array([[δs**2,c12],[c12,δt**2]])
    #Cinv = np.linalg.inv(C)
    
    #χ2 = d.T @ (Cinv @ d)
    
    χ2 = (δs**2*(t-t_mean)**2+δt**2*(s-s_mean)**2-2*c12*(t-t_mean)*(s-s_mean))/(δs**2*δt**2-c12**2)
    
    set_dir = 'data_'+'GTHDM'+strgl5+'-'+strga+'/'
    path_file_old = Path('./'+set_dir+'/GTHDM'+strgl5+'-'+strga+'-STU_PDG.csv')
    
    if path_file_old.exists():
        TableTot_STU_old = pd.read_csv('./'+set_dir+'/GTHDM'+strgl5+'-'+strga+'-STU_PDG.csv')
    
        s_old, t_old = TableTot_STU_old['S-parameter (1-loop BSM)'],TableTot_STU_old['T-parameter (1-loop BSM)']
        χ2old = (δs**2*(t_old-t_mean)**2+δt**2*(s_old-s_mean)**2-2*c12*(t_old-t_mean)*(s_old-s_mean))/(δs**2*δt**2-c12**2)
        
        χ2min = np.min([np.min(χ2),np.min(χ2old)])
    else:
        χ2min = np.min(χ2)
    
    Δχ = χ2-χ2min
    CL = stats.chi2.cdf(Δχ, 2)
    
    res = np.where(CL<0.95,False,True)
    
    return res

def collider_const(HiggsB):
    return np.where(HiggsB,False,True)

def signals_const(chisq):
    N_observables = hggfcs.signals.observableCount()
    N_d_freedom = N_observables-N_parameters
    
    if fcs.alignment:
        strga = 'A'
    else:
        strg = str(fcs.non_alignment_max)
        strga = 'NA'+strg
        
    if fcs.small_l5:
        strgl5 = '331lk'
    else:
        strgl5 = ''
    
    set_dir = 'data_'+'GTHDM'+strgl5+'-'+strga+'/'
    path_file_old = Path('./'+set_dir+'/GTHDM'+strgl5+'-'+strga+'-Collid_PDG.csv')
    
    if path_file_old.exists():
        TableTot_STU_Collid_old = pd.read_csv('./'+set_dir+'/GTHDM'+strgl5+'-'+strga+'-Collid_PDG.csv')
    
        χ2old = TableTot_STU_Collid_old['HiggsS']
        
        χ2min = np.min([np.min(chisq),np.min(χ2old)])
    else:
        χ2min = np.min(chisq)
    
    Δχ = chisq-χ2min
    CL = stats.chi2.cdf(np.array(Δχ), N_d_freedom)
    
    return np.where(CL<0.95,False,True)

def calculate_lambda(DtFrame):
    lamb = []
    lambtree = []

    sino = np.sin(fcs.beta(DtFrame['tanb'])-fcs.alpha(DtFrame['cosa']))
    
    if fcs.alignment:
        THDM2 = anyBSM('GTHDMII', scheme_name = 'OSalignment')
    else:
        THDM2 = anyBSM('GTHDMII', scheme_name = 'OS')

    prog = 1
    tot = DtFrame.shape[0]
    for i in DtFrame.index:
        if not fcs.alignment and sino[i]==1.0:
            THDM2.load_renormalization_scheme('OSalignment')
            
        progress_bar(prog, tot)
        
        WmassStd = 'MWm'
        THDM2.setparameters({'Mh2': DtFrame.at[i,'mH'], 'MAh2': DtFrame.at[i,'mA'], 'MHm2': DtFrame.at[i,'mHpm'], 'TanBeta': DtFrame.at[i,'tanb'], 'SinBmA': sino[i],'M': DtFrame.at[i,'M'], 'rLam6': DtFrame.at[i,'l6'], 'rLam7': DtFrame.at[i,'l7']}) #Define new mass in anyBSM
        THDM2.progress=False
        THDM2.warnSSSS=False
        dic = THDM2.lambdahhh()
        lamb.append(-np.real(dic['total'])/fcs.Gammahhh_treelevel(0, 0))  #Recalculate lambda
        lambtree.append(-np.real(dic['treelevel'])/fcs.Gammahhh_treelevel(0, 0))  #Recalculate lambda
    
        prog+=1
    return lamb, lambtree

def calculate_quartics(DtFrame):
    c93 = qtcp.C_93(fcs.alpha(DtFrame['cosa']), fcs.beta(DtFrame['tanb']), DtFrame['mH'], DtFrame['l5'])
    c94 = qtcp.C_94(fcs.alpha(DtFrame['cosa']), fcs.beta(DtFrame['tanb']), DtFrame['mH'], DtFrame['mA'], DtFrame['l5'])
    c102 = qtcp.C_102(fcs.alpha(DtFrame['cosa']), fcs.beta(DtFrame['tanb']), DtFrame['mH'], DtFrame['mA'], DtFrame['l5'])
    c123 = qtcp.C_123(fcs.alpha(DtFrame['cosa']), fcs.beta(DtFrame['tanb']), DtFrame['mH'], DtFrame['l5'])
    c140 = qtcp.C_140(fcs.alpha(DtFrame['cosa']), fcs.beta(DtFrame['tanb']), DtFrame['mH'], DtFrame['l5'])
    
    return [c93,c94,c102,c123,c140]

def perturbative_unitarity_const(c):
    cmax = 8*np.pi
    res = np.where(np.abs(c)<cmax,False,True)
    
    return res

def calculate_eigenvalues(DtFrame):
    if fcs.alignment:
        THDM2 = anyBSM('GTHDMII', scheme_name = 'OSalignment')
    else:
        THDM2 = anyBSM('GTHDMII', scheme_name = 'OS')
        
    sino = np.sin(fcs.beta(DtFrame['tanb'])-fcs.alpha(DtFrame['cosa']))
    
    a0=[]
        
    THDM2.progress=False
    THDM2.warnSSSS=False
    
    WmassStd = 'MWm'
    for i in DtFrame.index:
        if not fcs.alignment and sino[i]==1.0:
            THDM2.load_renormalization_scheme('OSalignment')
            
        a0.append(THDM2.eigSSSS(parameters={'Mh2': DtFrame.at[i,'mH'], 'MAh2': DtFrame.at[i,'mA'], 'MHm2': DtFrame.at[i,'mHpm'], 'TanBeta': DtFrame.at[i,'tanb'], 'SinBmA': sino[i],'M': DtFrame.at[i,'M'], 'rLam6': DtFrame.at[i,'l6'], 'rLam7': DtFrame.at[i,'l7']}))

    return a0

def perturbative_unitarity_const_a0(a0):
    a0max = 0.5
    res = np.where(np.abs(a0)<a0max,False,True)
    
    return res

#%%                         Function to call

def main_module(N_points):
    s1 = time()

    Table = fcs.find_random_points(int(N_points))

    sb = np.sin(fcs.beta(Table['tanb']))
    cb = np.cos(fcs.beta(Table['tanb']))
    sa = np.sin(fcs.alpha(Table['cosa']))
    ca = Table['cosa']

    mh = fcs.mhSM
    v = fcs.v

    m122 = fcs.m122M(Table['M'], sb, cb)
    l1 = fcs.lamb1(Table['mH'], mh, m122, ca, sa, sb, cb, v, Table['l6'], Table['l7'])
    l2 = fcs.lamb2(Table['mH'], mh, m122, ca, sa, sb, cb, v, Table['l6'], Table['l7'])
    l3 = fcs.lamb3(Table['mH'], mh, Table['mHpm'], m122, ca, sa, sb, cb, v, Table['l6'], Table['l7'])
    l4 = fcs.lamb4(Table['mA'], Table['mHpm'], m122, sb, cb, v, Table['l6'], Table['l7'])
    l5 = fcs.lamb5(Table['mA'], m122, sb, cb, v, Table['l6'], Table['l7'])

    Table1 = pd.DataFrame(np.array([m122,l1,l2,l3,l4,l5]).T,columns=['m122','l1','l2','l3','l4','l5'])

    Table = pd.concat([Table,Table1],axis=1)
    Table1 = None
    e1 = time()
    print('Duration - Setting up the points: %f' %(e1-s1))

    #%%                                 Analysis

    ###     Perturbativity tests

    s2 = time()
    TableP = Table

    for i in range(1,6):
        cnd = perturbativity_bounds(TableP['l'+str(i)])
        TableP = TableP.drop(TableP[cnd].index)

    ###     Stability tests

    cnd = stability_bounds([Table['l'+str(i)] for i in range(1,6)])
    TableStab = Table.drop(Table[cnd].index)

    ###     Testing both

    cnd = stability_bounds([TableP['l'+str(i)] for i in range(1,6)])
    TableTot = TableP.drop(TableP[cnd].index)
    TableTot.reset_index()

    Table = None
    TableP = None
    TableStab = None
    l1 = None
    l2 = None
    l3 = None
    l4 = None
    l5 = None
    sa = None
    ca = None
    sb = None
    cb = None

    gc.collect()
    e2 = time()

    print('Duration - Theoretical bounds: %f' %(e2-s2))

    #%%                         Calculate lambda

    s3 = time()
    
    lamb, lambtree = calculate_lambda(TableTot)
    sino = np.sin(fcs.beta(TableTot['tanb'])-fcs.alpha(TableTot['cosa']))
    
    # Using that sinBmA ~ 1-x²/2
    kappa_kan_x = fcs.Gammahhh_oneloop(np.sqrt(2*(1-sino)), TableTot['M'], TableTot['mH'], TableTot['mA'], TableTot['mHpm'])/fcs.Gammahhh_treelevel(0, 0)
    kappa_kan = fcs.Gammahhh_oneloop_cos(fcs.beta(TableTot['tanb'])-fcs.alpha(TableTot['cosa']), fcs.beta(TableTot['tanb']), TableTot['M'], TableTot['mH'], TableTot['mA'], TableTot['mHpm'])/fcs.Gammahhh_treelevel(0, 0)
        
    TableTot = pd.concat([TableTot,pd.DataFrame({'kappa': lamb, 'kappa-tree': lambtree, 'kappa-kan-x': kappa_kan_x, 'kappa-kan': kappa_kan})],axis=1)

    e3 = time()
    print('Duration - Calculating kappa: %f' %(e3-s3))

    #%%                                 Calculate S,T,U & collider constraints

    s4 = time()
    Sp_in = TableTot.T.loc[['l1','l2','l3','l4','l5','l6','l7','m122','tanb']].T
    Sp_in['m122'] = -Sp_in['m122'] # Different convention from SPheno.
    Sp_in['l1'] = Sp_in['l1']/2 # Different convention from SPheno.
    Sp_in['l2'] = Sp_in['l2']/2 # Different convention from SPheno.
    Sp_in = Sp_in.rename({'m122': 'm12'},axis=1) # Fixed name with SPheno.

    TotalSP = Spheno_Higgstools_calc_(Sp_in)

    #%%                                 Impose bounds from STU and Collider

    STU = TotalSP.T.loc[['S-parameter (1-loop BSM)','T-parameter (1-loop BSM)','U-parameter (1-loop BSM)','HiggsB','HiggsS']].T
    STU['S-parameter (1-loop BSM)']=pd.to_numeric(STU['S-parameter (1-loop BSM)'],errors='coerce')
    STU['T-parameter (1-loop BSM)']=pd.to_numeric(STU['T-parameter (1-loop BSM)'],errors='coerce')
    STU['U-parameter (1-loop BSM)']=pd.to_numeric(STU['U-parameter (1-loop BSM)'],errors='coerce')
    STU['HiggsS']=pd.to_numeric(STU['HiggsS'],errors='coerce')
    STU=STU.drop(index=0).set_index(TableTot.index)
    TableTot = pd.concat([TableTot,STU],axis=1)

    cnd = STU_constraint(TableTot['S-parameter (1-loop BSM)'],TableTot['T-parameter (1-loop BSM)'],TableTot['U-parameter (1-loop BSM)'])
    TableTot_STU = TableTot.drop(TableTot[cnd].index)

    cnd = collider_const(TableTot_STU['HiggsB'])
    TableTot_STU_Collid = TableTot_STU.drop(TableTot_STU[cnd].index)
    e4 = time()
    print('Duration - STU & Collider bounds: %f' %(e4-s4))

    #%%                                 Impose bounds from HiggsSignals

    cnd = signals_const(np.array(TableTot_STU_Collid['HiggsS'],dtype=float))
    TableTot_STU_Collid = TableTot_STU_Collid.drop(TableTot_STU_Collid[cnd].index)

    #%%                                 Impose bounds from BSG

    s5 = time()
    cnd = bsg.Constraints_BSG(np.array(TableTot_STU_Collid['tanb'],dtype=float), np.array(TableTot_STU_Collid['mHpm'],dtype=float))
    TableTot_STU_Collid_BSG = TableTot_STU_Collid.drop(TableTot_STU_Collid[cnd].index)
    e5 = time()
    print('Duration - BSG bounds: %f' %(e5-s5))
    
    #%%                                 Impose perturbative unitarity bounds
    
    s6 = time()
    aux = pd.DataFrame(np.abs(np.array(calculate_quartics(TableTot_STU_Collid_BSG))).T,columns=['c93','c94','c102','c123','c140'])
    #aux=aux.drop(index=0).set_index(TableTot_STU_Collid_BSG.index)
    TableTot_STU_Collid_BSG=TableTot_STU_Collid_BSG.reset_index(drop=True)
    TableTot_STU_Collid_BSG_unit = pd.concat([TableTot_STU_Collid_BSG,aux],axis=1)

    # for cs in ['c93','c94','c102','c123','c140']:
    #     cnd = spr.perturbative_unitarity_const(TableTot_STU_Collid_BSG_unit[cs])
    #     TableTot_STU_Collid_BSG_unit = TableTot_STU_Collid_BSG_unit.drop(TableTot_STU_Collid_BSG_unit[cnd].index)

    TableTot_STU_Collid_BSG_unit = pd.concat([TableTot_STU_Collid_BSG_unit,pd.DataFrame(np.array(calculate_eigenvalues(TableTot_STU_Collid_BSG_unit)).T,columns=['a0'])],axis=1)

    cnd = perturbative_unitarity_const_a0(TableTot_STU_Collid_BSG_unit['a0'])
    TableTot_STU_Collid_BSG_unit = TableTot_STU_Collid_BSG_unit.drop(TableTot_STU_Collid_BSG_unit[cnd].index)
    e6 = time()
    
    print('Duration - PU bounds: %f' %(e6-s6))

    return TableTot, TableTot_STU, TableTot_STU_Collid, TableTot_STU_Collid_BSG, TableTot_STU_Collid_BSG_unit