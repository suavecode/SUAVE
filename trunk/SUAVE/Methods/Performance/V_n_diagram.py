## @ingroup Methods-Performance
# V_n_diagram.py
#
# Created:  Nov 2018, S. Karpuk 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave Imports
import SUAVE
from SUAVE.Core import Data
from SUAVE.Core import Units

from SUAVE.Analyses.Mission.Segments.Conditions import Aerodynamics,Numerics
#from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_max_lift_coeff

# package imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#  Compute a V-n diagram
# ----------------------------------------------------------------------

## @ingroup Methods-Performance
def V_n_diagram(vehicle,analyses,weight,altitude,delta_ISA,FARflag):
    
    """ Computes a V-n diagram for a given aircraft and given regulations for ISA conditions

    Source:
    S. Gudmundsson "General Aviation Aircraft Design: Applied Methods and Procedures", Butterworth-Heinemann; 1 edition
    CFR FAR Part 23: https://www.ecfr.gov/cgi-bin/text-idx?SID=0e6a13c7c1de7f501d0eb0a4d71418bd&mc=true&tpl=/ecfrbrowse/Title14/14cfr23_main_02.tpl
    CFR FAR Part 25: https://www.ecfr.gov/cgi-bin/text-idx?tpl=/ecfrbrowse/Title14/14cfr25_main_02.tpl

    Inputs:
    analyses.base.atmosphere               [SUAVE data type]
    vehicle.
      mass_properties.takeoff              [kg]
      reference_area                       [m^2]
      maximum_lift_coefficient             [Unitless]
      chords.mean_aerodynamic      [meter]

    Outputs:
    airspeeds, Mach, load_factors                   [m]

    Properties Used:
    N/A
    """        
    
    # ==============================================
    # Unpack
    # ==============================================
    atmo = analyses.configs.base.atmosphere

    for wing in vehicle.wings: 
        reference_area  = vehicle.reference_area 
        Cmac            = wing.chords.mean_aerodynamic
    
    for mission in analyses.missions:
        Vc = mission.segments.cruise.air_speed

    CLa = 5.25
    category_tag = vehicle.category
    minimum_lift_coefficient = -1.0
    vehicle.minimum_lift_coefficient = minimum_lift_coefficient
    
    # ==============================================
    # Computing atmospheric conditions
    # ==============================================
    atmo_values = atmo.compute_values(altitude,delta_ISA)
    SL_atmo_values = atmo.compute_values(0,delta_ISA)
    conditions  = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()

    rho = atmo_values.density
    sea_level_rho = SL_atmo_values.density
    a = atmo_values.speed_of_sound
    sea_level_gravity = atmo.planet.sea_level_gravity

    # ==============================================
    # Determining vehicle maximum lift coefficient
    # ==============================================
    try:   # aircraft maximum lift informed by user
        maximum_lift_coefficient = vehicle.maximum_lift_coefficient
    except:
        # Using semi-empirical method for maximum lift coefficient calculation
        from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_max_lift_coeff

        # Condition to CLmax calculation: 300KTAS @ specified altitude, ISA
        conditions.freestream=Data()
        conditions.freestream.density   = atmo_values.density
        conditions.freestream.dynamic_viscosity = atmo_values.dynamic_viscosity
        conditions.freestream.velocity  = 300. * Units.knots
        try:
            maximum_lift_coefficient, maximum_lift_coefficient_flaps, induced_drag_high_lift = compute_max_lift_coeff(vehicle,conditions)
            vehicle.maximum_lift_coefficient = maximum_lift_coefficient
        except:
            raise ValueError, "Maximum lift coefficient calculation error. Please, check inputs"
        
    # ============================================================================
    # Convert all terms to English (Used for FAR) and remove elements from arrays
    # ============================================================================
    altitude = altitude / Units.ft
    rho = rho[0,0] / Units['slug/ft**3']
    sea_level_rho = sea_level_rho[0,0] / Units['slug/ft**3']
    density_ratio = (rho/sea_level_rho)**0.5
    a = a[0,0] / Units['ft/s']
    sea_level_gravity = sea_level_gravity / Units['ft/s**2']
    weight = weight / Units['slug'] * sea_level_gravity
    reference_area = reference_area / Units['ft**2']
    Cmac = Cmac / Units.ft
    wing_loading = weight / reference_area
    maximum_lift_coefficient = maximum_lift_coefficient[0,0]
    
    load_factors_pos = np.zeros(shape=(5));     load_factors_neg = np.zeros(shape=(5));
    load_factors_pos[1] = 1;    load_factors_neg[1] = -1;
    airspeeds_pos = np.zeros(shape=(5));        airspeeds_neg = np.zeros(shape=(5))
        
    # ================================================
    # Establish limit maneuver load factors n+ and n- 
    # ================================================ 
    # CFR Part 25
    # Positive and negative limits
    if FARflag == 25:   
        load_factors_pos[2] = 2.1 + 24000/(weight+10000)
        if load_factors_pos[2] < 2.5:
            load_factors_pos[2] = 2.5
        elif load_factors_pos[2] > 3.8:
            load_factors_pos[2] = 3.8
        print('minimum positive limit load factor =  '+ str(load_factors_pos[2]))
        print('maximum positive limit load factor =  3.8')
        cond = raw_input('Would you like to increase it? (y/n): ')
        if cond == 'y':
            load_factors_pos[2] = raw_input('input positive limit maneuver load factor (n <= 3.8): ')
            
        # Negative limit      
        load_factors_neg[2] = raw_input('input negative limit maneuver load factor (n <= -1): ')

    elif FARflag == 23:
        if category_tag == 'normal' or category_tag == 'commuter':
            load_factors_pos[2] = 2.1 + 24000/(weight+10000)
            if load_factors_pos[2] < 2.5:
                load_factors_pos[2] = 2.5
            elif load_factors_pos[2] > 3.8:
                load_factors_pos[2] = 3.8
            print('minimum positive limit load factor =  '+ str(load_factors_pos[2]))
            print('maximum positive limit load factor =  3.8')
            cond = raw_input('Would you like to increase it? (y/n): ')
            if cond == 'y':
                load_factors_pos[2] = raw_input('input positive limit maneuver load factor (n <= 3.8): ')
            load_factors_neg[2] = -0.4 * load_factors_pos[2]
        elif category_tag == 'utility':
            load_factors_pos[2] = raw_input('input positive limit maneuver load factor (n >= 4.4): ')
            load_factors_neg[2] = -0.4 * load_factors_pos[2]
        elif category_tag == 'acrobatic':
            load_factors_pos[2] = raw_input('input positive limit maneuver load factor (n >= 6.0): ')
            load_factors_neg[2] = -0.5 * load_factors_pos[2]
        else:
            print('Check the category_tag input. The parameter was not found')
       
    else:
        print('Check the FARflag input. The parameter was not found')
        
    # ================================================
    # Computing critical speeds (Va, Vc, Vb, Vd, Vs1)
    # ================================================
    print("max_lift" + str(maximum_lift_coefficient))
    # Calculate Stall and Maneuver speeds
    airspeeds_pos, airspeeds_neg = stall_maneuver_speeds(airspeeds_pos,airspeeds_neg, weight, rho, reference_area, \
                                                         maximum_lift_coefficient, minimum_lift_coefficient,\
                                                         load_factors_pos, load_factors_neg)
        
    # convert speeds to KEAS for future calculations
    airspeeds_pos =  airspeeds_pos * Units['ft/s'] / Units.knots * density_ratio
    airspeeds_neg =  airspeeds_neg * Units['ft/s'] / Units.knots * density_ratio
        
    # Gust speeds between Vb and Vc(EAS) and minimum Vc
    miu = 2 * wing_loading / (rho * Cmac * CLa * sea_level_gravity)
    Kg = 0.88 * miu / (5.3 + miu)
    if FARflag == 25:
        if altitude < 15000:
            Uref = (-0.0008 * altitude + 56);       Uref_dive = 0.5 * Uref
        else:
            Uref = (-0.0005142 * altitude + 51.7133);       Uref_dive = 0.5 * Uref
        # Minimum Cruise speed Vc_min
        coefs=[1, -Uref * (2.64 + (Kg*CLa*airspeeds_pos[1]**2)/(498*wing_loading)), 1.72424*Uref]
        Vc1 = max(np.roots(coefs))
        
    elif FARflag == 23:
        if altitude < 20000:
            Uref = 50;      Uref_dive = 25;
        else:
            Uref = (-0.0008333 * altitude + 66.67);    Uref_dive = (-0.0004167 * altitude + 33.334);  
        # Minimum Cruise speed Vc_min
        if category_tag == 'acrobatic':
            Vc1 = 36 * wing_loaing**0.5
            if wing_loading >= 20:
                Vc1 = (-0.0925*wing_loading+37.85)* wing_loading**0.5
        else:
            Vc1 = 33 * wing_loading**0.5
            if wing_loading >= 20:       
                Vc1 = (-0.055*wing_loading+34.1)* wing_loading**0.5

    # input Cruise speed
    print('Cruise speed conditions: ')
    print('Vc > ' + str(Vc1) + 'KEAS')
    print('Vc >= Va, Va = ' + str(airspeeds_pos[2]) + 'KEAS')
    Vc = input('Input cruise speed Vc: ')

    # Dive speed
    if FARflag == 25:        
        airspeeds_pos[4] = 1.25 * Vc
    elif FARflag == 23:
        if category_tag == 'acrobatic':
            airspeeds_pos[4] = 1.55 * Vc1
        elif category_tag == 'utility':
            airspeeds_pos[4] = 1.5 * Vc1
        else:
            airspeeds_pos[4] = 1.4 * Vc1
       
    Vd = airspeeds_pos[4]
    airspeeds_pos[3] = airspeeds_pos[4]
    airspeeds_neg[4] = airspeeds_pos[4]
    if FARflag == 25:
        airspeeds_neg[3] = Vc
    else:
        airspeeds_neg[3] = airspeeds_pos[4]
    
    # complete initail load factors
    load_factors_pos[4] = 0;    load_factors_neg[4] = 0;           
    load_factors_pos[3] = load_factors_pos[2]
    load_factors_neg[3] = load_factors_neg[2]

    # Create Stall lines
    Num_of_points = 20                                            # number of points for the stall line
    upper_bound = 2;  lower_bound = 1;
    load_factors_pos, airspeeds_pos = stall_line(load_factors_pos, airspeeds_pos, upper_bound, lower_bound, Num_of_points, \
                                                 density_ratio, rho, maximum_lift_coefficient, weight, reference_area)
    load_factors_neg, airspeeds_neg = stall_line(load_factors_neg, airspeeds_neg, upper_bound, lower_bound, Num_of_points, \
                                                 density_ratio, rho, minimum_lift_coefficient, weight, reference_area)

    #print(airspeeds_pos)
    #print(airspeeds_neg)
    #print(load_factors_pos)
    #print(load_factors_neg)

    # ==============================================
    # ==============================================
    # Determine Gust loads
    # ==============================================
    # =========================================================================================================
    airspeeds_pos, load_factors_pos, gust_load_factors_pos = Gust_loads(airspeeds_pos, load_factors_pos, Kg, CLa, weight, \
                                                                        wing_loading, reference_area, rho, density_ratio, \
                                                                        maximum_lift_coefficient, Vc, Vd, Uref, Uref_dive, \
                                                                        Num_of_points, FARflag)

    airspeeds_neg, load_factors_neg, gust_load_factors_neg = Gust_loads(airspeeds_neg, load_factors_neg, Kg, CLa, weight, \
                                                                        wing_loading, reference_area, rho, density_ratio, \
                                                                        minimum_lift_coefficient, Vc, Vd, -Uref, -Uref_dive, \
                                                                        Num_of_points, FARflag)
        
    
    # =========================================================================================================    
    #print(airspeeds_neg)
    #print(load_factors_neg)
    # ==============================================
    # Post-processing the V-n diagram
    # ==============================================
    fig, ax = plt.subplots()
    ax.fill(airspeeds_pos, load_factors_pos, c='b', alpha=0.3)
    ax.fill(airspeeds_neg, load_factors_neg, c='b', alpha=0.3)
    ax.plot(airspeeds_pos, load_factors_pos, c='b')
    ax.plot(airspeeds_neg, load_factors_neg, c='b')
    
    ax.plot([0, Vc],[1,gust_load_factors_pos[2]],'--', c='r')
    ax.plot([0, Vd],[1,gust_load_factors_pos[3]],'--', c='r')
    ax.plot([0, Vc],[1,gust_load_factors_neg[2]],'--', c='r')
    ax.plot([0, Vd],[1,gust_load_factors_neg[3]],'--', c='r')
    print(vehicle)
    ax.set_xlabel('Airspeed, KEAS')
    ax.set_ylabel('Load Factor')
    ax.set_title(vehicle._base.tag + '  Weight=' + str(round(weight)) + 'lb  ' + ' Altitude=' + str(altitude) + 'ft ')
    ax.grid()
    plt.show()
    return 


#===================================================================================================================
#====================
#USEFUL FUNCTIONS
#===================================================================================================================
def stall_maneuver_speeds(airspeeds_pos, airspeeds_neg, weight, rho, reference_area, maximum_lift_coefficient, \
                          minimum_lift_coefficient, load_factors_pos, load_factors_neg):
    # Stall speeds
    airspeeds_pos[1] = (2 * weight / (rho * reference_area * maximum_lift_coefficient)) ** 0.5
    airspeeds_neg[1] = (2 * weight / (rho * reference_area * abs(minimum_lift_coefficient))) ** 0.5
    airspeeds_pos[0] = airspeeds_pos[1]
    airspeeds_neg[0] = airspeeds_neg[1]

    # Maneuver speeds
    airspeeds_pos[2] = airspeeds_pos[1] * load_factors_pos[2] ** 0.5
    airspeeds_neg[2] = (2 * weight * abs(load_factors_neg[2]) / (rho * reference_area * \
                                                                 abs(minimum_lift_coefficient))) ** 0.5

    return airspeeds_pos, airspeeds_neg

def stall_line(load_factors, airspeeds, upper_bound, lower_bound, Num_of_points, density_ratio, \
               rho, lift_coefficient, weight, reference_area):
    
    delta = (airspeeds[upper_bound] - airspeeds[lower_bound]) / (Num_of_points+1)     # Step size
    for i in range(Num_of_points):           
        coef = lower_bound+i+1
        airspeeds = np.concatenate((airspeeds[:coef], [airspeeds[lower_bound] + (i+1) * delta], airspeeds[coef:]))
        Vtas = airspeeds[coef] / density_ratio * Units.knots / Units['ft/s']
        if load_factors[1] > 0:
            nl = 0.5 * rho*Vtas**2 * reference_area * lift_coefficient / weight
        else:
            nl = -0.5 * rho * Vtas**2 * reference_area * abs(lift_coefficient) / weight
        load_factors = np.concatenate((load_factors[:coef], [nl], load_factors[coef:]))
    
    return load_factors, airspeeds

def Gust_loads(airspeeds, load_factors, Kg, CLa, weight, wing_loading, reference_area, rho, density_ratio, \
               lift_coefficient, Vc, Vd, Uref, Uref_dive, Num_of_points, FARflag):

    gust_load_factors = np.zeros(shape=(4));    
    gust_load_factors[0] = 1;        

    # Cruise speed Gust loads at Va and Vc
    gust_load_factors[1] = 1 + Kg*CLa*airspeeds[Num_of_points+2]*Uref/(498*wing_loading)
    gust_load_factors[2] = 1 + Kg*CLa*Vc*Uref/(498*wing_loading)

    print(gust_load_factors, load_factors)
    print(gust_load_factors[2], load_factors[Num_of_points+3])
    
    # ======================================================
    # Intersection between positive cruise gust load and Va
    # ======================================================
    if abs(gust_load_factors[1]) > abs(load_factors[Num_of_points+2]):
        print('Intersection between positive cruise gust load and Va')
        sea_level_rho = rho / density_ratio**2
        coefs = [709.486*sea_level_rho*lift_coefficient, -Kg*Uref*CLa, -498*wing_loading]
        V_inters = max(np.roots(coefs))
        load_factor_inters = 1 + Kg*CLa*V_inters*Uref/(498*wing_loading)

        airspeeds = np.concatenate((airspeeds[:(Num_of_points+3)], [V_inters], airspeeds[(Num_of_points+3):]))
        load_factors = np.concatenate((load_factors[:(Num_of_points+3)], [load_factor_inters], load_factors[(Num_of_points+3):]))
        
        # continue stall lines
        Num_of_points_ext = 5;
        upper_bound = Num_of_points+3;     lower_bound = Num_of_points+2;
        load_factors, airspeeds = stall_line(load_factors, airspeeds, upper_bound, lower_bound, Num_of_points_ext, \
                                                     density_ratio, rho, lift_coefficient, weight, reference_area)        
        Num_of_points = Num_of_points + Num_of_points_ext + 1

        # insert the cruise speed Vc in the positive load factor line
        if load_factors[1] > 0 or FARflag == 23:
            airspeeds = np.concatenate((airspeeds[:(Num_of_points+3)], [Vc], airspeeds[(Num_of_points+3):]))
            load_factors = np.concatenate((load_factors[:(Num_of_points+3)], [gust_load_factors[2]], \
                                           load_factors[(Num_of_points+3):]))

    # ======================================================================
    # Intersection between positive cruise gust load and maximum load at Vc
    # ======================================================================
    elif abs(gust_load_factors[2]) > abs(load_factors[Num_of_points+3]):
        V_inters = 498*(load_factors[Num_of_points+2]-1)*wing_loading/(Kg * Uref * CLa)
        airspeeds = np.concatenate((airspeeds[:(Num_of_points+3)], [V_inters], airspeeds[(Num_of_points+3):]))
        load_factors = np.concatenate((load_factors[:(Num_of_points+3)], [load_factors[Num_of_points+2]], \
                                       load_factors[(Num_of_points+3):]))
        Num_of_points =Num_of_points + 1
        
        if load_factors[1] > 0 or FARflag == 23:
            airspeeds = np.concatenate((airspeeds[:(Num_of_points+3)], [Vc], airspeeds[(Num_of_points+3):]))
            load_factors = np.concatenate((load_factors[:(Num_of_points+3)], [gust_load_factors[2]], \
                                           load_factors[(Num_of_points+3):]))
        print(airspeeds, gust_load_factors, load_factors)

        
    #==========================
    # Dive speed Gust loads Vd
    #==========================
    gust_load_factors[3] = 1 + Kg*CLa*Vd*Uref_dive/(498*wing_loading)
    if abs(gust_load_factors[2]) > abs(load_factors[len(load_factors)-2]):
        if abs(gust_load_factors[3]) > abs(load_factors[len(load_factors)-2]):
            load_factors[len(load_factors)-2] = gust_load_factors[3]
            if load_factors[1] > 0:
                airspeeds, load_factors = gust_dive_speed_intersection(load_factors, gust_load_factors, airspeeds, \
                                                                       len(load_factors)-2, Vc, Vd, FARflag)                
        else:
            airspeeds, load_factors = gust_dive_speed_intersection(load_factors, gust_load_factors, airspeeds, \
                                                                   len(load_factors)-2, Vc, Vd, FARflag)
            
    return airspeeds, load_factors, gust_load_factors

def gust_dive_speed_intersection(load_factors, gust_load_factors, airspeeds, element_num, Vc, Vd, FARflag):
    if FARflag == 25 and load_factors[1] > 0:
        V_inters = (gust_load_factors[3]*Vc+(load_factors[element_num]-gust_load_factors[2])*Vd)\
                   /(gust_load_factors[3]-gust_load_factors[2]+load_factors[element_num])
        load = load_factors[element_num]*(V_inters-Vd)/(Vc-Vd)
    else:       
        V_inters = (load_factors[element_num] - gust_load_factors[2])*(Vd-Vc)\
                    /(gust_load_factors[3] - gust_load_factors[2])+Vc
        load = load_factors[element_num]

    airspeeds = np.concatenate((airspeeds[:(element_num)], [V_inters], airspeeds[(element_num):]))
    load_factors = np.concatenate((load_factors[:(element_num)], [load], \
                                        load_factors[(element_num):]))    

    return airspeeds, load_factors

