"""
Petro-elastic model routines
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seis2rock.elastic as el
import seis2rock.bounds as bd

from seis2rock.elastic import Elastic, ElasticLimited
from seis2rock.solid import Matrix, Rock
from seis2rock.fluid import _batze_wang_brine, Brine, Gas, Oil, Fluid
from seis2rock.gassmann import Gassmann
from seis2rock.units import *
from seis2rock.wavelets import *
from seis2rock.Logs import *



def pem_seis2rock(phi, vsh, sw,
        pres=2.41e7, #Pressure (Pa)  
        temp=50, # Temperature [C] 
        sal=10000, # Ssalinity ppm
        oilgrav=20, # Oil gravity
        gasgrav=0.9, # Gas gravity
        gor=160
         ):
        
    """Petro-elastic modelling"""
    
    
    shape = phi.shape
    phi, vsh, sw = phi.ravel(), vsh.ravel(), sw.ravel()
    
    mat = Matrix({'shale': {'k': 37.6e9, 'mu': 44.6e9, 'rho': g_cm3_to_kg_m3(2.65), 'frac':vsh.ravel()},
                  'sand': {'k': 20.9e9, 'mu': 30.6e9,'rho': g_cm3_to_kg_m3(2.58), 'frac':1-vsh.ravel()}})

    k_min_avg=mat.k
    mu_min_avg=mat.mu
    rho_min_avg=mat.rho

    kmin= np.copy(k_min_avg)
    gmin= np.copy(mu_min_avg)
    
    #Poisson Ratio of the grain material
    numin=(3*kmin-2*gmin)/(2*(3*kmin+gmin))

    #Coordination number
    c = 4.46+9.7*(0.384-phi)**0.4
    
    # Effective Moduli (Hertz-Mindlin Model)
    k_eff_HM=((c**2*(1-phi)**2*gmin**2)/(18*np.pi**2*(1-numin)**2)*pres)**(1/3);
    mu_eff_HM=(5-4*numin)/(5*(2-numin))*((3*c**2*(1-phi)**2*gmin**2) / (2*np.pi**2*(1-numin)**2)*pres)**(1/3)

    # Brine, Oil, and Gas
    wat = Brine(temp, pres/10**6, sal)
    oil = Oil(temp, pres/10**6, oilgrav, gasgrav, gor)
    gas = Gas(temp, pres/10**6, gasgrav)

    # Build the fluid
    fluid0 = Fluid({'water': (wat, sw.ravel()),
            'oil': (oil, 1-sw.ravel()),
            'gas': (gas, np.zeros_like(sw.ravel()))})
    k_fluid=fluid0.k
    rho_fluid=fluid0.rho
    
    # K, mu, and rho saturated (Gassmann)
    kdry = k_eff_HM
    kmatrix = kmin
    ksat = kdry + (1 - kdry / kmatrix) ** 2 / (
            phi / k_fluid + (1 - phi) /
            kmatrix - kdry / kmatrix ** 2)
    musat = mu_eff_HM
    rhosat = rho_min_avg * (1 - phi) + rho_fluid * phi

    #Vp and Vs saturated
    vp_sat = ((ksat + 4./3.*musat)/rhosat)**0.5
    vs_sat =  (musat/rhosat)**0.5
    
    # Reshape inputs
    phi, vsh, sw = phi.reshape(shape), vsh.reshape(shape), sw.reshape(shape)

    return vp_sat.reshape(shape), vs_sat.reshape(shape), rhosat.reshape(shape)