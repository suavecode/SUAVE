## @ingroup Methods-Center_of_Gravity
# compute_fuel_center_of_gravity_longitudinal_range.py
#
# Created:  Sep 2018, T. MacDonald
# Modified: Oct 2018, T. MacDonald
#           Jan 2019, T. MacDonald

import numpy as np

## @ingroup Methods-Center_of_Gravity
def plot_cg_map(masses,cg_mins,cg_maxes,empty_mass=0,empty_cg=0):
    """Plot possible longitudinal cg positions for the fuel (or full vehicle
    is empty mass is given)
    
    Assumptions:
    None

    Source:
    N/A

    Inputs:
    masses     [kg]
    cg_mins    [m]
    cg_maxes   [m]
    empty_mass [kg]
    empty_cg   [m]

    Outputs:
    A plot

    Properties Used:
    N/A
    """    
    
    ylabel_string = 'Fuel Mass (kg)'
    
    if empty_mass != 0:
        cg_maxes = (cg_maxes*masses+empty_cg*empty_mass)/(masses+empty_mass)
        cg_mins  = (cg_mins*masses+empty_cg*empty_mass)/(masses+empty_mass)
        masses   = masses+empty_mass
        ylabel_string = 'Total Mass (kg)'
    
    import pylab as plt

    fig = plt.figure("Available Fuel CG Distribution",figsize=(8,6))
    axes = plt.gca()
    axes.plot(cg_maxes,masses,'g-') 
    axes.plot(cg_mins,masses,'b-')
    
    axes.set_xlabel('CG Position (m)')
    axes.set_ylabel(ylabel_string)
    axes.set_title('Available Fuel CG Distribution')
    axes.grid(True)  
    
    plt.show()
    
    return

## @ingroup Methods-Center_of_Gravity
def compute_fuel_center_of_gravity_longitudinal_range(vehicle):
    """Computes the possible longitudinal center of gravity given
    a set of fuel tanks (includes fuel weight only)

    Assumptions:
    Fuel tanks are only in the fuselage/wings

    Source:
    N/A

    Inputs:
    vehicle.wings.*.Fuel_Tanks.mass_properties.
      center_of_gravity       [m]
      fuel_mass_when_full     [kg]
    vehicle.fuselages.*.Fuel_Tanks.mass_properties.
      center_of_gravity       [m]
      fuel_mass_when_full     [kg]

    Outputs:
    fuel_masses               [kg] (these are arrays spanning the possible masses)
    min_cg                    [m]
    max_cg                    [m]

    Properties Used:
    N/A
    """  
    
    fuel_tanks = []
    
    for wing in vehicle.wings:
        for tank in wing.Fuel_Tanks:
            fuel_tanks.append(tank)    
    
    for fuse in vehicle.fuselages:
        for tank in fuse.Fuel_Tanks:
            fuel_tanks.append(tank)    
                    
    fuel_tanks.sort(key=lambda x: x.mass_properties.center_of_gravity[0][0])
    
    tank_cgs    = np.zeros(len(fuel_tanks))
    tank_masses = np.zeros(len(fuel_tanks))
    
    for i,tank in enumerate(fuel_tanks):
        tank_cgs[i]    = tank.mass_properties.center_of_gravity[0][0]
        tank_masses[i] = tank.mass_properties.fuel_mass_when_full
    
    max_mass = np.sum(tank_masses)
    
    fuel_masses = np.linspace(1e-6,max_mass)
    min_cg      = np.zeros_like(fuel_masses)
    max_cg      = np.zeros_like(fuel_masses)
    
    tank_masses_front_to_back = tank_masses
    tank_masses_back_to_front = np.flip(tank_masses,0)
    tank_cgs_front_to_back    = tank_cgs
    tank_cgs_back_to_front    = np.flip(tank_cgs,0)
    
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