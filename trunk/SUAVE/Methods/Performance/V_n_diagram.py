## @ingroup Methods-Performance
# V_n_diagram.py
#
# Created:  Nov 2018, S. Karpuk
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave Imports
import SUAVE
from SUAVE.Core import Data
from SUAVE.Core import Units

from SUAVE.Analyses.Mission.Segments.Conditions import Aerodynamics,Numerics
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_max_lift_coeff
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations import datcom

# package imports
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#  Compute a V-n diagram
# ----------------------------------------------------------------------

## @ingroup Methods-Performance
def V_n_diagram(vehicle,analyses,weight,altitude,delta_ISA):
    
    """ Computes a V-n diagram for a given aircraft and given regulations for ISA conditions

    Source:
    S. Gudmundsson "General Aviation Aircraft Design: Applied Methods and Procedures", Butterworth-Heinemann; 1 edition
    CFR FAR Part 23: https://www.ecfr.gov/cgi-bin/text-idx?SID=0e6a13c7c1de7f501d0eb0a4d71418bd&mc=true&tpl=/ecfrbrowse/Title14/14cfr23_main_02.tpl
    CFR FAR Part 25: https://www.ecfr.gov/cgi-bin/text-idx?tpl=/ecfrbrowse/Title14/14cfr25_main_02.tpl

    Inputs:
    analyses.base.atmosphere               [SUAVE data type]
    vehicle.
      reference_area                       [m^2]
      maximum_lift_coefficient             [Unitless]
      minimum_lift_coefficient             [Unitless]
      chords.mean_aerodynamic              [m]
      envelope.FARpart_number              [Unitless]
        limit_loads.positive               [Unitless]
        limit_loads.negative               [Unitless]
        cruise_mach                        [Unitless]
    weight                                 [kg]
    altitude                               [m]
    delta_ISA                              [deg C]

    Outputs:
    V_n_data

    Properties Used:
    N/A

    Description:
    The script creates an aircraft V-n diagram based on the input parameters specified by the user.
    Depending on the certification flag, an appropriate diagram, output and log files are created.
    """        

    #------------------------
    # Create a log - file
    #------------------------
    flog = open("V_n_diagram_" + vehicle.tag + ".log","w")
    
    print('Running the V-n diagram calculation...')
    flog.write('Running the V-n diagram calculation...\n')
    flog.write('Aircraft: ' + vehicle.tag + '\n')
    flog.write('Category: ' + vehicle.envelope.category + '\n')
    flog.write('FAR certification: Part ' + str(vehicle.envelope.FAR_part_number) + '\n\n')
    
    # ----------------------------------------------
    # Unpack
    # ----------------------------------------------
    flog.write('Unpacking the input and calculating required inputs...\n')
    FAR_part_number = vehicle.envelope.FAR_part_number
    atmo            = analyses.atmosphere
    Mc              = vehicle.envelope.cruise_mach

    for wing in vehicle.wings: 
        reference_area  = vehicle.reference_area 
        Cmac            = wing.chords.mean_aerodynamic

    for envelope in vehicle:
        pos_limit_load  = vehicle.envelope.limit_loads.positive
        neg_limit_load  = vehicle.envelope.limit_loads.negative

    category_tag = vehicle.envelope.category
    
    # ----------------------------------------------
    # Computing atmospheric conditions
    # ----------------------------------------------
    atmo_values       = atmo.compute_values(altitude,delta_ISA)
    SL_atmo_values    = atmo.compute_values(0,delta_ISA)
    conditions        = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()

    rho               = atmo_values.density
    sea_level_rho     = SL_atmo_values.density
    sea_level_gravity = atmo.planet.sea_level_gravity
    Vc                = Mc * (1.4 * 287 * atmo_values.temperature) ** 0.5
    
    # ------------------------------
    # Computing lift-curve slope
    # ------------------------------  
    CLa = datcom(vehicle.wings.main_wing, np.array([Mc]))
    CLa = CLa[0]

    # -----------------------------------------------------------
    # Determining vehicle minimum and maximum lift coefficients
    # -----------------------------------------------------------
    try:   # aircraft maximum lift informed by user
        maximum_lift_coefficient = vehicle.maximum_lift_coefficient
    except:
        # Condition to CLmax calculation: 0.333 * Vc @ specified altitude, ISA
        conditions.freestream                   = Data()
        conditions.freestream.density           = atmo_values.density
        conditions.freestream.dynamic_viscosity = atmo_values.dynamic_viscosity
        conditions.freestream.velocity          = 0.333 * Vc
        try:
            max_lift_coefficient, induced_drag_high_lift \
                                             = compute_max_lift_coeff(vehicle,conditions)
            maximum_lift_coefficient         = max_lift_coefficient[0][0]
            vehicle.maximum_lift_coefficient = maximum_lift_coefficient
            
        except:
            raise ValueError("Maximum lift coefficient calculation error. Please, check inputs")
        
    try:    # aircraft minimum lift informed by user
        minimum_lift_coefficient = vehicle.minimum_lift_coefficient 
    except:
        raise ValueError("The value not found. Specify minimum lift coefficient")
             
    # -----------------------------------------------------------------------------
    # Convert all terms to English (Used for FAR) and remove elements from arrays
    # -----------------------------------------------------------------------------
    altitude          = altitude / Units.ft
    rho               = rho[0,0] / Units['slug/ft**3']
    sea_level_rho     = sea_level_rho[0,0] / Units['slug/ft**3']
    density_ratio     = (rho/sea_level_rho)**0.5
    sea_level_gravity = sea_level_gravity / Units['ft/s**2']
    weight            = weight / Units['slug'] * sea_level_gravity
    reference_area    = reference_area / Units['ft**2']
    Cmac              = Cmac / Units.ft
    wing_loading      = weight / reference_area
    Vc                = Vc / Units['ft/s']
    
    load_factors_pos    = np.zeros(shape=(5));
    load_factors_neg    = np.zeros(shape=(5));
    load_factors_pos[1] =  1;
    load_factors_neg[1] = -1;
    airspeeds_pos       = np.zeros(shape=(5));
    airspeeds_neg       = np.zeros(shape=(5))
        
    # --------------------------------------------------
    # Establish limit maneuver load factors n+ and n- 
    # --------------------------------------------------
    flog.write('Establish limit maneuver load factors n+ and n-...\n')
    # CFR Part 25
    # Positive and negative limits
    if FAR_part_number == 25:
        # Positive limit
        flog.write('    Estimating n+...\n')
        load_factors_pos[2] = 2.1 + 24000 / (weight + 10000)
        if load_factors_pos[2] < 2.5:
            flog.write('        Defined positive limit load factor < 2.5. Setting 2.5...\n')
            load_factors_pos[2] = 2.5
        elif load_factors_pos[2] < pos_limit_load:
            load_factors_pos[2] = pos_limit_load
        elif load_factors_pos[2] > 3.8:
            flog.write('        Defined positive limit load factor > 3.8. Setting 3.8...\n')
            load_factors_pos[2] = 3.8

        # Negative limit
        flog.write('    Estimating n-...\n')
        load_factors_neg[2] = neg_limit_load
        if load_factors_neg[2] > -1:
            flog.write('        Negative limit load factor magnitude is too small. Setting to -1...\n')
            load_factors_neg[2] = -1

    elif FAR_part_number == 23:
        if category_tag == 'normal' or category_tag == 'commuter':
            # Positive limit
            flog.write('    Estimating n+...\n')
            load_factors_pos[2] = 2.1 + 24000 / (weight + 10000)
            if load_factors_pos[2] < 2.5:
                flog.write('        Defined positive limit load factor < 2.5. Setting 2.5...\n')
                load_factors_pos[2] = 2.5
            elif load_factors_pos[2] < pos_limit_load:
                load_factors_pos[2] = pos_limit_load
            elif load_factors_pos[2] > 3.8:
                flog.write('        Defined positive limit load factor > 3.8. Setting 3.8...\n')
                load_factors_pos[2] = 3.8

            # Negative limit
            flog.write('    Estimating n-...\n')
            load_factors_neg[2] = -0.4 * load_factors_pos[2]
            
        elif category_tag == 'utility':
            # Positive limit
            flog.write('    Estimating n+...\n')
            load_factors_pos[2] =  pos_limit_load
            if load_factors_pos[2] < 4.4:
                 flog.write('        Defined positive limit load factor < 4.4. Setting 4.4...\n')
                 load_factors_pos[2] = 4.4
                 
            # Negative limit
            flog.write('    Estimating n-...\n')
            load_factors_neg[2] = -0.4 * load_factors_pos[2]
            
        elif category_tag == 'acrobatic':
            # Positive limit
            load_factors_pos[2] = pos_limit_load
            if load_factors_pos[2] < 6.0:
                flog.write('        Defined positive limit load factor < 6.0. Setting 6.0...\n')
                load_factors_pos[2] = 6.0

            # Negative limit    
            load_factors_neg[2] = -0.5 * load_factors_pos[2]
            
        else:
            raise ValueError("Check the category_tag input. The parameter was not found")
       
    else:
        raise ValueError("Check the FARflag input. The parameter was not found")

    # Check input of the limit load
    if abs(neg_limit_load) > abs(load_factors_neg[2]):
        load_factors_neg[2] = neg_limit_load

    #----------------------------------------
    # Generate a V-n diagram data structure
    #----------------------------------------
    V_n_data                          = Data()
    V_n_data.limit_loads              = Data()
    V_n_data.limit_loads.dive         = Data()
    V_n_data.load_factors             = Data()
    V_n_data.gust_load_factors        = Data()
    V_n_data.Vb_load_factor           = Data()
    V_n_data.airspeeds                = Data()
    V_n_data.Vs1                      = Data()
    V_n_data.Va                       = Data()
    V_n_data.Vb                       = Data()
    V_n_data.load_factors.positive    = load_factors_pos
    V_n_data.load_factors.negative    = load_factors_neg
    V_n_data.airspeeds.positive       = airspeeds_pos
    V_n_data.airspeeds.negative       = airspeeds_neg
    V_n_data.Vc                       = Vc
    V_n_data.weight                   = weight
    V_n_data.wing_loading             = wing_loading
    V_n_data.altitude                 = altitude
    V_n_data.density                  = rho
    V_n_data.density_ratio            = density_ratio
    V_n_data.reference_area           = reference_area
    V_n_data.maximum_lift_coefficient = maximum_lift_coefficient
    V_n_data.minimum_lift_coefficient = minimum_lift_coefficient
    V_n_data.limit_loads.positive     = load_factors_pos[2]
    V_n_data.limit_loads.negative     = load_factors_neg[2]
    
    # --------------------------------------------------
    # Computing critical speeds (Va, Vc, Vb, Vd, Vs1)
    # --------------------------------------------------
    flog.write('Computing critical speeds (Va, Vc, Vb, Vd, Vs1)...\n')
    
    # Calculate Stall and Maneuver speeds
    flog.write('    Computing Vs1 and Va...\n')
    stall_maneuver_speeds(V_n_data)  
        
    # convert speeds to KEAS for future calculations 
    convert_keas(V_n_data)

    # unpack modified airspeeds 
    airspeeds_pos = V_n_data.airspeeds.positive
    airspeeds_neg = V_n_data.airspeeds.negative
    Vc            = V_n_data.Vc
    Va_pos        = V_n_data.Va.positive 
    Va_neg        = V_n_data.Va.negative
    Va_pos        = V_n_data.Va.positive 
    Va_neg        = V_n_data.Va.negative 

    flog.write('    Checking Vc wrt Va...\n')
    if Va_neg > Vc and Va_neg > Va_pos:
        flog.write('        Negative Va > Vc. Setting Vc = 1.15 * Va...\n')
        Vc = 1.15 * Va_neg
    elif Va_pos > Vc and Va_neg < Va_pos:
        flog.write('        Positive Va > Vc. Setting Vc = 1.15 * Va...\n')
        Vc = 1.15 * Va_pos
    
    # Gust speeds between Vb and Vc (EAS) and minimum Vc
    miu = 2 * wing_loading / (rho * Cmac * CLa * sea_level_gravity)
    Kg  = 0.88 * miu / (5.3 + miu)
          
    if FAR_part_number == 25:
        if altitude < 15000:
            Uref_cruise = (-0.0008 * altitude + 56)
            Uref_rough  = Uref_cruise
            Uref_dive   = 0.5 * Uref_cruise
        else:
            Uref_cruise = (-0.0005142 * altitude + 51.7133)
            Uref_rough  = Uref_cruise
            Uref_dive   = 0.5 * Uref_cruise
            
        # Minimum Cruise speed Vc_min
        coefs = [1, -Uref_cruise * (2.64 + (Kg * CLa * airspeeds_pos[1]**2)/(498 * wing_loading)), 1.72424 * Uref_cruise**2 - airspeeds_pos[1]**2]
        Vc1   = max(np.roots(coefs))
        
    elif FAR_part_number == 23:           
        if altitude < 20000:
            Uref_cruise = 50;
            Uref_dive   = 25;
        else:
            Uref_cruise = (-0.0008333 * altitude + 66.67);
            Uref_dive = (-0.0004167 * altitude + 33.334);

        if category_tag == 'commuter':
            if altitude < 20000:
                Uref_rough = 66
            else:
                Uref_rough = -0.000933 * altitude + 84.667
        else:
            Uref_rough = Uref_cruise
                       
        # Minimum Cruise speed Vc_min
        if category_tag == 'acrobatic':
            Vc1 = 36 * wing_loading**0.5
            if wing_loading >= 20:
                Vc1 = (-0.0925 * wing_loading + 37.85) * wing_loading **0.5
        else:
            Vc1 = 33 * wing_loading**0.5
            if wing_loading >= 20:       
                Vc1 = (-0.055 * wing_loading + 34.1) * wing_loading **0.5

    # checking input Cruise speed
    flog.write('    Checking Vc wrt Vcmin... \n')
    if Vc1 > Vc:
        flog.write('        Specified cruise speed is less than the minimum required. Setting th minimum required value...\n')
        Vc = Vc1

    # Dive speed
    flog.write('    Computing Vd...\n')
    if FAR_part_number == 25:        
        airspeeds_pos[4] = 1.25 * Vc
    elif FAR_part_number == 23:
        if category_tag == 'acrobatic':
            airspeeds_pos[4] = 1.55 * Vc1
            if wing_loading > 20:
                airspeeds_pos[4] = (-0.0025 * wing_loading + 1.6) * Vc1
        elif category_tag == 'utility':
            airspeeds_pos[4] = 1.5 * Vc1
            if wing_loading > 20:
                airspeeds_pos[4] = (-0.001875 * wing_loading + 1.5375) * Vc1           
        else:
            airspeeds_pos[4] = 1.4 * Vc1
            if wing_loading > 20:
                airspeeds_pos[4] = (-0.000625 * wing_loading + 1.4125) * Vc1
            
        if airspeeds_pos[4] < 1.15 * Vc:
            flog.write('        Based on min Vc, Vd is too close Vc. Setting Vd = 1.15 Vc...\n')
            airspeeds_pos[4] = 1.15 * Vc
   
    Vd               = airspeeds_pos[4]
    airspeeds_pos[3] = airspeeds_pos[4]
    airspeeds_neg[3] = Vc
    airspeeds_neg[4] = airspeeds_pos[4]
    
    # complete initial load factors
    load_factors_pos[4] = 0
    load_factors_neg[4] = 0
    if category_tag == 'acrobatic' or category_tag == 'utility':
        load_factors_neg[4] = -1
        
    load_factors_pos[3] = load_factors_pos[2]
    load_factors_neg[3] = load_factors_neg[2]

    # add parameters to the data structure
    V_n_data.load_factors.positive    = load_factors_pos
    V_n_data.load_factors.negative    = load_factors_neg
    V_n_data.airspeeds.positive       = airspeeds_pos
    V_n_data.airspeeds.negative       = airspeeds_neg
    V_n_data.Vd                       = Vd
    V_n_data.Vc                       = Vc

    #------------------------
    # Create Stall lines
    #------------------------
    flog.write('Creating stall lines...\n')
    Num_of_points = 20                                            # number of points for the stall line
    upper_bound = 2;
    lower_bound = 1;
    stall_line(V_n_data, upper_bound, lower_bound, Num_of_points, 1)
    stall_line(V_n_data, upper_bound, lower_bound, Num_of_points, 2)

    # ----------------------------------------------
    # Determine Gust loads
    # ----------------------------------------------
    flog.write('Calculating gust loads...\n')
    V_n_data.gust_data           = Data()
    V_n_data.gust_data.airspeeds = Data()
    
    V_n_data.gust_data.airspeeds.rough_gust  = Uref_rough
    V_n_data.gust_data.airspeeds.cruise_gust = Uref_cruise
    V_n_data.gust_data.airspeeds.dive_gust   = Uref_dive
    
    gust_loads(category_tag, V_n_data, Kg, CLa, Num_of_points, FAR_part_number, 1)
    gust_loads(category_tag, V_n_data, Kg, CLa, Num_of_points, FAR_part_number, 2)

    #----------------------------------------------------------------
    # Finalize the load factors for acrobatic and utility aircraft
    #----------------------------------------------------------------
    if category_tag == 'acrobatic' or category_tag == 'utility':
        V_n_data.airspeeds.negative    = np.append(V_n_data.airspeeds.negative, Vd)
        V_n_data.load_factors.negative = np.append(V_n_data.load_factors.negative, 0)

    # ----------------------------------------------
    # Post-processing the V-n diagram
    # ----------------------------------------------
    flog.write('Post-Processing...\n')
    V_n_data.limit_loads.positive = max(V_n_data.load_factors.positive)
    V_n_data.limit_loads.negative = min(V_n_data.load_factors.negative)
    
    post_processing(category_tag, Uref_rough, Uref_cruise, Uref_dive, V_n_data, vehicle)

    print('Calculation complete...')
    flog.write('Calculation complete\n')
    flog.close()
  
    return V_n_data


#------------------------------------------------------------------------------------------------------
# USEFUL FUNCTIONS
#------------------------------------------------------------------------------------------------------

def stall_maneuver_speeds(V_n_data):

    """ Computes stall and maneuver speeds for positive and negative halves of the the V-n diagram

    Source:
    S. Gudmundsson "General Aviation Aircraft Design: Applied Methods and Procedures", Butterworth-Heinemann; 1 edition

    Inputs:
    V_n_data.
        airspeeds.positive                      [kts]
            negative                            [kts]
        weight                                  [lb]
        density                                 [slug/ft**3]
        density_ratio                           [Unitless]
        reference_area                          [ft**2]
        maximum_lift_coefficient                [Unitless]
        minimum_lift_coefficient                [Unitless]
        load_factors.positive                   [Unitless]
        load_factors.negative                   [Unitless]

    Outputs:
    V_n_data.
        airspeeds.positive                      [kts]
            negative                            [kts]
        Vs1.positive                            [kts]       
            negative                            [kts]
        Va.positive                             [kts]
            negative                            [kts]
        load_factors.positive                   [Unitless]
            negative                            [Unitless]                 

    Properties Used:
    N/A

    Description:   
    """     

    # Unpack
    airspeeds_pos    = V_n_data.airspeeds.positive
    airspeeds_neg    = V_n_data.airspeeds.negative
    weight           = V_n_data.weight
    rho              = V_n_data.density
    reference_area   = V_n_data.reference_area
    max_lift_coef    = V_n_data.maximum_lift_coefficient
    min_lift_coef    = V_n_data.minimum_lift_coefficient
    load_factors_pos = V_n_data.load_factors.positive
    load_factors_neg = V_n_data.load_factors.negative
    
    # Stall speeds
    airspeeds_pos[1] = (2 * weight / (rho * reference_area * max_lift_coef)) ** 0.5
    airspeeds_neg[1] = (2 * weight / (rho * reference_area * abs(min_lift_coef))) ** 0.5
    airspeeds_pos[0] = airspeeds_pos[1]
    airspeeds_neg[0] = airspeeds_neg[1]

    # Maneuver speeds
    airspeeds_pos[2] = airspeeds_pos[1] * load_factors_pos[2] ** 0.5
    airspeeds_neg[2] = (2 * weight * abs(load_factors_neg[2]) / (rho * reference_area * \
                                                                 abs(min_lift_coef))) ** 0.5
    
    # Pack
    V_n_data.airspeeds.positive           = airspeeds_pos
    V_n_data.airspeeds.negative           = airspeeds_neg
    V_n_data.load_factors.positive        = load_factors_pos
    V_n_data.load_factors.negative        = load_factors_neg
    V_n_data.Vs1.positive                 = airspeeds_pos[1]
    V_n_data.Vs1.negative                 = airspeeds_neg[1]
    V_n_data.Va.positive                  = airspeeds_pos[2]
    V_n_data.Va.negative                  = airspeeds_neg[2]
    
#------------------------------------------------------------------------------------------------------------

def stall_line(V_n_data, upper_bound, lower_bound, Num_of_points, sign_flag):
    
    """ Calculates Stall lines of positive and negative halves of the V-n diagram

    Source:
    S. Gudmundsson "General Aviation Aircraft Design: Applied Methods and Procedures", Butterworth-Heinemann; 1 edition

    Inputs:
    V_n_data.
        airspeeds.positive                  [kts]
            negative                        [kts]
        load_factors.positive               [Unitless]
            negative                        [Unitless]
        weight                              [lb]
        density                             [slug/ft**3]
        density_ratio                       [Unitless]
        lift_coefficient                    [Unitless]
        reference_area                      [ft**2]
    lower_bound                             [Unitless]
    Num_of_points                           [Unitless]
    sign_flag                               [Unitless]

    Outputs:
    V_n_data.
        load_factors.positive               [Unitless]
            negative                        [Unitless]
        airspeeds.positive                  [kts]
            negative                        [kts]

    Properties Used:
    N/A

    Description:   
    """  
    # Unpack
    weight          = V_n_data.weight
    reference_area  = V_n_data.reference_area
    density_ratio   = V_n_data.density_ratio
    rho             = V_n_data.density
    
    if sign_flag == 1:
        load_factors     = V_n_data.load_factors.positive
        airspeeds        = V_n_data.airspeeds.positive
        lift_coefficient = V_n_data.maximum_lift_coefficient

    elif sign_flag == 2:
        load_factors     = V_n_data.load_factors.negative
        airspeeds        = V_n_data.airspeeds.negative
        lift_coefficient = V_n_data.minimum_lift_coefficient
    
    delta = (airspeeds[upper_bound] - airspeeds[lower_bound]) / (Num_of_points + 1)     # Step size
    for i in range(Num_of_points):       
        coef      = lower_bound + i + 1
        airspeeds = np.concatenate((airspeeds[:coef], [airspeeds[lower_bound] + (i + 1) * delta], airspeeds[coef:]))
        Vtas      = airspeeds[coef] / density_ratio * Units.knots / Units['ft/s']      
        if load_factors[1] > 0:
            nl = 0.5 * rho * Vtas**2 * reference_area * lift_coefficient / weight
        else:
            nl = -0.5 * rho * Vtas**2 * reference_area * abs(lift_coefficient) / weight
            
        load_factors = np.concatenate((load_factors[:coef], [nl], load_factors[coef:]))

    # Pack
    if sign_flag == 1:
        V_n_data.load_factors.positive     = load_factors
        V_n_data.airspeeds.positive        = airspeeds

    elif sign_flag == 2:
        V_n_data.load_factors.negative     = load_factors
        V_n_data.airspeeds.negative        = airspeeds
    
    return 
#--------------------------------------------------------------------------------------------------------------

def gust_loads(category_tag, V_n_data, Kg, CLa, Num_of_points, FAR_part_number, sign_flag):

    """ Calculates airspeeds and load factors for gust loads and modifies the V-n diagram

    Source:
    S. Gudmundsson "General Aviation Aircraft Design: Applied Methods and Procedures", Butterworth-Heinemann; 1 edition

    Inputs:
    V_n_data.
        airspeeds.positive                      [kts]
            negative                            [kts]
        load_factors.positive                   [Unitless]
            negative                            [Unitless]
        limit_loads.positive                    [Unitless]
            negative                            [Unitless]
        weight                                  [lb]
        wing_loading                            [lb/ft**2]
        reference_area                          [ft**2]
        density                                 [slug/ft**3]
        density_ratio                           [Unitless]
        lift_coefficient                        [Unitless]
        Vc                                      [kts]
        Vd                                      [kts]
    Uref_rough                                  [ft/s]
    Uref_cruise                                 [ft/s]
    Uref_dive                                   [ft/s]
    Num_of_points                               [Unitless]
    sign_flag                                   [Unitless]
  
    
    Outputs:
    V_n_data.
        airspeeds.positive                      [kts]
            negative                            [kts]
        load_factors.positive                   [Unitless]
            .negative                           [Unitless]
        gust_load_factors.positive              [Unitless]
            negative                            [Unitless]

    Properties Used:
    N/A

    Description:
    The function calculates the gust-related load factors at critical speeds (Va, Vc, Vd). Then, if the load factors exceed
    the standart diagram limits, the diagram is modified to include the new limit loads.
    For more details, refer to S. Gudmundsson "General Aviation Aircraft Design: Applied Methods and Procedures"
    """

    # Unpack
    weight          = V_n_data.weight
    wing_loading    = V_n_data.wing_loading
    reference_area  = V_n_data.reference_area
    density         = V_n_data.density
    density_ratio   = V_n_data.density_ratio
    Vc              = V_n_data.Vc
    Vd              = V_n_data.Vd
    
    if sign_flag == 1:
        airspeeds        = V_n_data.airspeeds.positive
        load_factors     = V_n_data.load_factors.positive
        lift_coefficient = V_n_data.maximum_lift_coefficient
        limit_load       = V_n_data.limit_loads.positive
        Uref_rough       = V_n_data.gust_data.airspeeds.rough_gust
        Uref_cruise      = V_n_data.gust_data.airspeeds.cruise_gust
        Uref_dive        = V_n_data.gust_data.airspeeds.dive_gust

    elif sign_flag == 2:
        airspeeds        = V_n_data.airspeeds.negative
        load_factors     = V_n_data.load_factors.negative
        lift_coefficient = V_n_data.minimum_lift_coefficient
        limit_load       = V_n_data.limit_loads.negative
        Uref_rough       = -V_n_data.gust_data.airspeeds.rough_gust
        Uref_cruise      = -V_n_data.gust_data.airspeeds.cruise_gust
        Uref_dive        = -V_n_data.gust_data.airspeeds.dive_gust
        

    gust_load_factors    = np.zeros(shape=(4));    
    gust_load_factors[0] = 1;
    
    # Cruise speed Gust loads at Va and Vc
    gust_load_factors[1] = 1 + Kg * CLa * airspeeds[Num_of_points+2] * Uref_rough / (498 * wing_loading)
    gust_load_factors[2] = 1 + Kg * CLa * Vc * Uref_cruise/(498 * wing_loading)

    # Intersection between cruise gust load and Va
    if abs(gust_load_factors[1]) > abs(limit_load):
        sea_level_rho      = density / density_ratio**2
        coefs              = [709.486 * sea_level_rho * lift_coefficient, -Kg * Uref_rough * CLa, -498 * wing_loading]
        V_inters           = max(np.roots(coefs))
        load_factor_inters = 1 + Kg * CLa * V_inters * Uref_rough / (498 * wing_loading)

        airspeeds    = np.concatenate((airspeeds[:(Num_of_points + 3)], [V_inters], airspeeds[(Num_of_points + 3):]))
        load_factors = np.concatenate((load_factors[:(Num_of_points + 3)], [load_factor_inters], load_factors[(Num_of_points + 3):]))

        # Pack
        if sign_flag == 1:
            V_n_data.airspeeds.positive          = airspeeds
            V_n_data.load_factors.positive       = load_factors
            V_n_data.gust_load_factors.positive  = gust_load_factors
            V_n_data.Vb_load_factor.positive     = load_factor_inters
            V_n_data.Vb.positive                 = V_inters           
            
        if sign_flag == 2:
            V_n_data.airspeeds.negative          = airspeeds
            V_n_data.load_factors.negative       = load_factors
            V_n_data.gust_load_factors.negative  = gust_load_factors
            V_n_data.Vb_load_factor.negative     = load_factor_inters
            V_n_data.Vb.negative                 = V_inters
        
        # continue stall lines
        Num_of_points_ext       = 5;
        upper_bound             = Num_of_points+3;
        lower_bound             = Num_of_points+2;
        stall_line(V_n_data, upper_bound, lower_bound, Num_of_points_ext, sign_flag)        
        Num_of_points = Num_of_points + Num_of_points_ext + 1

        # Unpack
        if sign_flag == 1:
            airspeeds        = V_n_data.airspeeds.positive
            load_factors     = V_n_data.load_factors.positive
            lift_coefficient = V_n_data.maximum_lift_coefficient

        elif sign_flag == 2:
            airspeeds        = V_n_data.airspeeds.negative
            load_factors     = V_n_data.load_factors.negative
            lift_coefficient = V_n_data.minimum_lift_coefficient

        
        # insert the cruise speed Vc in the positive load factor line
        if load_factors[1] > 0:
            airspeeds    = np.concatenate((airspeeds[:(Num_of_points + 3)], [Vc], airspeeds[(Num_of_points + 3):]))
            load_factors = np.concatenate((load_factors[:(Num_of_points + 3)], [gust_load_factors[2]], \
                                           load_factors[(Num_of_points + 3):]))

    # Intersection between cruise gust load and maximum load at Vc
    elif abs(gust_load_factors[2]) > abs(limit_load):
        V_inters      = 498 * (load_factors[Num_of_points + 2] - 1) * wing_loading / (Kg * Uref_cruise * CLa)
        airspeeds     = np.concatenate((airspeeds[:(Num_of_points + 3)], [V_inters], airspeeds[(Num_of_points + 3):]))
        load_factors  = np.concatenate((load_factors[:(Num_of_points + 3)], [limit_load], \
                                       load_factors[(Num_of_points + 3):]))
        Num_of_points = Num_of_points + 1

        if load_factors[1] > 0:
            airspeeds    = np.concatenate((airspeeds[:(Num_of_points + 3)], [Vc], airspeeds[(Num_of_points + 3):]))
            load_factors = np.concatenate((load_factors[:(Num_of_points + 3)], [gust_load_factors[2]], \
                                           load_factors[(Num_of_points + 3):]))
        else:
            load_factors[len(airspeeds)-2] = gust_load_factors[2]

        
    # Dive speed Gust loads Vd
    gust_load_factors[3] = 1 + Kg * CLa * Vd * Uref_dive / (498 * wing_loading)

    # Resolve the upper half of the dive section
    if load_factors[1] > 0: 
        if abs(gust_load_factors[2]) > abs(limit_load):   
            if abs(gust_load_factors[3]) > abs(limit_load):
                load_factors[len(load_factors) - 2] = gust_load_factors[3]               
            else:
                airspeeds, load_factors = gust_dive_speed_intersection(category_tag, load_factors, gust_load_factors, airspeeds, \
                                                                       len(load_factors)-2, Vc, Vd)

    # Resolve the lower half of the dive section
    else:
        if gust_load_factors[3] < load_factors[len(load_factors) - 1]:
            airspeeds    = np.concatenate((airspeeds[:(len(load_factors) - 1)], [airspeeds[(len(load_factors) - 1)]], \
                                           airspeeds[(len(load_factors) - 1):]))
            load_factors = np.concatenate((load_factors[:(len(load_factors) - 1)], [gust_load_factors[3]], \
                                           load_factors[(len(load_factors) - 1):]))
            
            if abs(gust_load_factors[2]) > abs(limit_load):
                load_factors[len(load_factors) - 2] = gust_load_factors[3]
            else:
                airspeeds, load_factors = gust_dive_speed_intersection(category_tag, load_factors, gust_load_factors, airspeeds, \
                                                                       len(load_factors)-2, Vc, Vd)

            V_n_data.limit_loads.dive.negative = load_factors[len(load_factors) - 2]
        else:
            V_n_data.limit_loads.dive.negative = load_factors[len(load_factors) - 1]

    # gusts load extension for gust lines at Vd
    gust_load_factors = np.append(gust_load_factors, 1 + Kg * CLa * (1.05 * Vd) * Uref_cruise/(498 * wing_loading))
    gust_load_factors = np.append(gust_load_factors, 1 + Kg * CLa * (1.05 * Vd) * Uref_dive/(498 * wing_loading))

    # guts load extension for gust lines at Vb and Vc for a rough gust
    if category_tag == 'commuter':
        gust_load_factors = np.append(gust_load_factors, 1 + Kg * CLa * (1.05 * Vd) * Uref_rough/(498 * wing_loading))
    else:
        gust_load_factors = np.append(gust_load_factors, 0)
      
    # Pack
    if sign_flag == 1:
        V_n_data.airspeeds.positive          = airspeeds
        V_n_data.load_factors.positive       = load_factors
        V_n_data.gust_load_factors.positive  = gust_load_factors
        V_n_data.limit_loads.dive.positive   = load_factors[len(load_factors) - 2]
        
    if sign_flag == 2:
        V_n_data.airspeeds.negative          = airspeeds
        V_n_data.load_factors.negative       = load_factors
        V_n_data.gust_load_factors.negative  = gust_load_factors
    
    return 
#------------------------------------------------------------------------------------------------------------------------

def gust_dive_speed_intersection(category_tag, load_factors, gust_load_factors, airspeeds, element_num, Vc, Vd):

    """ Calculates intersection between the general V-n diagram and the gust load for Vd

    Source:
    S. Gudmundsson "General Aviation Aircraft Design: Applied Methods and Procedures", Butterworth-Heinemann; 1 edition

    Inputs:
    load_factors                                [Unitless]
    gust_load_factors                           [Unitless]
    airspeeds                                   [kts]
    Vc                                          [kts]
    Vd                                          [kts]
    element_num                                 [Unitless]
    
    Outputs:
    airspeeds                                    [kts]
    load_factors                                 [Unitless]

    Properties Used:
    N/A

    Description:
    A specific function for CFR FAR Part 25 regulations. For negative loads, the general curve must go linear from a specific
    load factor at Vc to zero at Vd. Gust loads may put a non-zero load factor at Vd, so an intersection between the old and
    the new curves is required
    """
    
    if load_factors[1] > 0:
        V_inters = (load_factors[element_num] - gust_load_factors[2]) * (Vd - Vc) \
                    / (gust_load_factors[3] - gust_load_factors[2]) + Vc
        load     = load_factors[element_num]
    else:
        if category_tag == 'acrobatic':          
            V_inters = ((gust_load_factors[3] + 1) * Vc + (min(load_factors) - gust_load_factors[2]) * Vd) / \
                       (gust_load_factors[3] - gust_load_factors[2] + min(load_factors) + 1)
            load     = (min(load_factors) + 1) / (Vc - Vd) * V_inters - ((min(load_factors) + 1) / (Vc - Vd) * Vd + 1)
        else:
            V_inters = (gust_load_factors[3] * Vc + (min(load_factors) - gust_load_factors[2]) * Vd) / \
                       (gust_load_factors[3] - gust_load_factors[2] + min(load_factors))
            load     = min(load_factors) * (V_inters - Vd) / (Vc - Vd)


    airspeeds    = np.concatenate((airspeeds[:(element_num)], [V_inters], airspeeds[(element_num):]))
    load_factors = np.concatenate((load_factors[:(element_num)], [load], \
                                        load_factors[(element_num):]))    

    return airspeeds, load_factors
#--------------------------------------------------------------------------------------------------------------------------

def post_processing(category_tag, Uref_rough, Uref_cruise, Uref_dive, V_n_data, vehicle):

    """ Plot graph, save the final figure, and create results output file

    Source:

    Inputs:
    V_n_data.
        airspeeds.positive                  [kts]
        airspeeds.negative                  [kts]
        Vc                                  [kts]
        Vd                                  [kts]
        Vs1.positive                        [kts]
            negative                        [kts]
        Va.positive                         [kts]
            negative                        [kts]
        load_factors.positive               [Unitless]
            negative                        [Unitless]
        gust_load_factors.positive          [Unitless]
            negative                        [Unitless]
        weight                              [lb]
        altitude                            [ft]
    vehicle._base.tag                       [Unitless]
    Uref_rough                              [ft/s]
    Uref_cruise                             [ft/s]
    Uref_dive                               [ft/s]

    Outputs:

    Properties Used:
    N/A

    Description:
    """

    # Unpack
    load_factors_pos        = V_n_data.load_factors.positive
    load_factors_neg        = V_n_data.load_factors.negative
    airspeeds_pos           = V_n_data.airspeeds.positive
    airspeeds_neg           = V_n_data.airspeeds.negative
    Vc                      = V_n_data.Vc
    Vd                      = V_n_data.Vd
    Vs1_pos                 = V_n_data.Vs1.positive
    Vs1_neg                 = V_n_data.Vs1.negative
    Va_pos                  = V_n_data.Va.positive
    Va_neg                  = V_n_data.Va.negative
    gust_load_factors_pos   = V_n_data.gust_load_factors.positive
    gust_load_factors_neg   = V_n_data.gust_load_factors.negative
    weight                  = V_n_data.weight
    altitude                = V_n_data.altitude

    #-----------------------------
    # Plotting the V-n diagram
    #-----------------------------
    fig, ax = plt.subplots()
    ax.fill(airspeeds_pos, load_factors_pos, c='b', alpha=0.3)
    ax.fill(airspeeds_neg, load_factors_neg, c='b', alpha=0.3)
    ax.plot(airspeeds_pos, load_factors_pos, c='b')
    ax.plot(airspeeds_neg, load_factors_neg, c='b')

    # Plotting gust lines
    ax.plot([0, Vc,1.05*Vd],[1,gust_load_factors_pos[2],gust_load_factors_pos[len(gust_load_factors_pos)-3]],'--', c='r', label = ('Gust ' + str(round(Uref_cruise)) + 'fps'))
    ax.plot([0, Vd,1.05*Vd],[1,gust_load_factors_pos[3],gust_load_factors_pos[len(gust_load_factors_pos)-2]],'--', c='g', label = ('Gust ' + str(round(Uref_dive)) + 'fps'))
    ax.plot([0, Vc,1.05*Vd],[1,gust_load_factors_neg[2],gust_load_factors_neg[len(gust_load_factors_neg)-3]],'--', c='r')
    ax.plot([0, Vd,1.05*Vd],[1,gust_load_factors_neg[3],gust_load_factors_neg[len(gust_load_factors_neg)-2]],'--', c='g')

    if category_tag == 'commuter':
        ax.plot([0, 1.05*Vd],[1,gust_load_factors_pos[len(gust_load_factors_pos)-1]],'--', c='m', label = ('Gust ' + str(round(Uref_rough)) + 'fps'))
        ax.plot([0, 1.05*Vd],[1,gust_load_factors_neg[len(gust_load_factors_neg)-1]],'--', c='m')

    # Formating the plot
    ax.set_xlabel('Airspeed, KEAS')
    ax.set_ylabel('Load Factor')
    ax.set_title(vehicle.tag + '  Weight=' + str(round(weight)) + 'lb  ' + ' Altitude=' + str(round(altitude)) + 'ft ')
    ax.legend()
    ax.grid()
    plt.savefig('Vn_diagram_'+ vehicle.tag + '.png')

    #---------------------------------
    # Creating results output file
    #---------------------------------
    fres = open("V_n_diagram_results_" + vehicle.tag +".dat","w")
    fres.write('V-n diagram summary\n')
    fres.write('-------------------\n')
    fres.write('Aircraft: ' + vehicle.tag + '\n')
    fres.write('category: ' + vehicle.envelope.category + '\n')
    fres.write('FAR certification: Part ' + str(vehicle.envelope.FAR_part_number) + '\n')
    fres.write('Weight = ' + str(round(weight)) + ' lb\n')
    fres.write('Altitude = ' + str(round(altitude)) + ' ft\n')
    fres.write('---------------------------------------------------------------\n\n')
    fres.write('Airspeeds: \n')
    fres.write('    Positive stall speed (Vs1)   = ' + str(round(Vs1_pos,1)) + ' KEAS\n')
    fres.write('    Negative stall speed (Vs1)   = ' + str(round(Vs1_neg,1)) + ' KEAS\n')
    fres.write('    Positive maneuver speed (Va) = ' + str(round(Va_pos,1))  + ' KEAS\n')
    fres.write('    Negative maneuver speed (Va) = ' + str(round(Va_neg,1))  + ' KEAS\n')
    fres.write('    Cruise speed (Vc)            = ' + str(round(Vc,1))      + ' KEAS\n')
    fres.write('    Dive speed (Vd)              = ' + str(round(Vd,1))      + ' KEAS\n')
    fres.write('Load factors: \n')
    fres.write('    Positive limit load factor (n+) = ' + str(round(max(load_factors_pos),2)) + '\n')
    fres.write('    Negative limit load factor (n-) = ' + str(round(min(load_factors_neg),2)) + '\n')
    fres.write('    Positive load factor at Vd      = ' + str(round(V_n_data.limit_loads.dive.positive,2)) + '\n')
    fres.write('    Negative load factor at Vd      = ' + str(round(V_n_data.limit_loads.dive.negative,2)) + '\n')
   
    return
#------------------------------------------------------------------------------------------------------------------------

def convert_keas(V_n_data):

    """ Convert speed to KEAS

    Source:

    Inputs:
    V_n_data.
        airspeeds.positive              [ft/s]
            negative                    [ft/s]
        Vc                              [ft/s]
        Va.positive                     [ft/s]
        Va.negative                     [ft/s]
        Vs1.negative                    [ft/s]
        Vs1.positive                    [ft/s]
        density_ratio                   [Unitless]

    Outputs:
    V_n_data.              
        airspeeds.positive              [kts]
            negative                    [kts]
        Vc                              [kts]
        Va.positive                     [kts]
        Va.negative                     [kts]
        Vs1.negative                    [kts]
        Vs1.positive                    [kts]

    Properties Used:
    N/A

    Description:
    """

    # Unpack
    airspeeds_pos = V_n_data.airspeeds.positive
    airspeeds_neg = V_n_data.airspeeds.negative
    density_ratio = V_n_data.density_ratio
    Vc            = V_n_data.Vc
    Va_pos        = V_n_data.Va.positive
    Va_neg        = V_n_data.Va.negative
    Vs1_neg       = V_n_data.Vs1.negative
    Vs1_pos       = V_n_data.Vs1.positive
    
    airspeeds_pos = airspeeds_pos * Units['ft/s'] / Units.knots * density_ratio
    airspeeds_neg = airspeeds_neg * Units['ft/s'] / Units.knots * density_ratio

    Vs1_pos       = Vs1_pos * Units['ft/s'] / Units.knots * density_ratio
    Vs1_neg       = Vs1_neg * Units['ft/s'] / Units.knots * density_ratio
    Va_pos        = Va_pos * Units['ft/s'] / Units.knots * density_ratio
    Va_neg        = Va_neg * Units['ft/s'] / Units.knots * density_ratio
    Vc            = Vc[0] * Units['ft/s'] / Units.knots * density_ratio
    Vc            = Vc[0]
    
    # Pack
    V_n_data.airspeeds.positive = airspeeds_pos
    V_n_data.airspeeds.negative = airspeeds_neg
    V_n_data.Vc                 = Vc
    V_n_data.Va.positive        = Va_pos
    V_n_data.Va.negative        = Va_neg
    V_n_data.Vs1.negative       = Vs1_neg
    V_n_data.Vs1.positive       = Vs1_pos
    
    
