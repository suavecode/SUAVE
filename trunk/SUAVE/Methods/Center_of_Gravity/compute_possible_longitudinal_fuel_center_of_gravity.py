## @ingroup Methods-Center_of_Gravity
# compute_possible_longitudinal_fuel_center_of_gravity.py
#
# Created:  Sep 2018, T. MacDonald

# ----------------------------------------------------------------------
#  Computer Mission Center of Gravity
# ----------------------------------------------------------------------

from SUAVE.Core import DataOrdered, Data
import numpy as np
from copy import copy

def plot_cg_map(masses,cg_mins,cg_maxes):
    
    import pylab as plt

    fig = plt.figure("Available Fuel CG Distribution",figsize=(8,6))
    axes = plt.gca()
    axes.plot(cg_maxes,masses,'g-') 
    axes.plot(cg_mins,masses,'b-')
    
    axes.set_xlabel('CG Position (m)')
    axes.set_ylabel('Fuel Mass (kg)')
    axes.set_title('Available Fuel CG Distribution')
    axes.grid(True)  
    
    plt.show()

## @ingroup Methods-Center_of_Gravity
def compute_possible_longitudinal_fuel_center_of_gravity(vehicle):
    """ 

    Assumptions:
    None

    Source:
    N/A

    Inputs:

    Outputs:

    Properties Used:
    N/A
    """  
    
    fuel_tanks = []
    
    if 'wings' in vehicle:
        for wing in vehicle.wings:
            if 'Fuel_Tanks' in wing:
                for tank in wing.Fuel_Tanks:
                    fuel_tanks.append(tank)    
    
    if 'fuselages' in vehicle:
        for fuse in vehicle.fuselages:
            if 'Fuel_Tanks' in fuse:
                for tank in fuse.Fuel_Tanks:
                    fuel_tanks.append(tank)    
                    
    fuel_tanks.sort(key=lambda x: x.mass_properties.center_of_gravity[0,0])
    
    tank_cgs    = np.zeros(len(fuel_tanks))
    tank_masses = np.zeros(len(fuel_tanks))
    
    for i,tank in enumerate(fuel_tanks):
        tank_cgs[i]    = tank.mass_properties.center_of_gravity[0,0]
        tank_masses[i] = tank.mass_properties.full_fuel_volume
        
    #tank_cgs = np.array([0,1,2])
    #tank_masses = np.array([1,1,1])
    
    max_mass = np.sum(tank_masses)
    
    fuel_masses = np.linspace(1e-6,max_mass)
    min_cg      = np.zeros_like(fuel_masses)
    max_cg      = np.zeros_like(fuel_masses)
    
    tank_masses_front_to_back = tank_masses
    tank_masses_back_to_front = np.flip(tank_masses)
    tank_cgs_front_to_back    = tank_cgs
    tank_cgs_back_to_front    = np.flip(tank_cgs)
    
    for j,mass in enumerate(fuel_masses):
        # find minimum
        remaining_mass = mass
        min_numer      = 0
        for i,tank_mass in enumerate(tank_masses_front_to_back):
            if remaining_mass == 0:
                break
            elif remaining_mass > tank_mass:
                min_numer += tank_cgs_front_to_back[i]*tank_mass
                remaining_mass -= tank_mass
            else:
                min_numer += tank_cgs_front_to_back[i]*remaining_mass
                remaining_mass = 0.
        min_cg[j] = min_numer/mass
        # find maximum
        remaining_mass = mass
        max_numer      = 0
        for i,tank_mass in enumerate(tank_masses_back_to_front):
            if remaining_mass == 0:
                break
            elif remaining_mass > tank_mass:
                max_numer += tank_cgs_back_to_front[i]*tank_mass
                remaining_mass -= tank_mass
            else:
                max_numer += tank_cgs_back_to_front[i]*remaining_mass
                remaining_mass = 0.
        max_cg[j] = max_numer/mass   
   
    return fuel_masses, min_cg, max_cg