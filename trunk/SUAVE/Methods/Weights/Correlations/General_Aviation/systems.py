## @ingroup Methods-Weights-Correlations-General_Aviation
# systems.py
# 
# Created:  Feb 2018, M. Vegh
# Modified: 


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units, Data

# ----------------------------------------------------------------------
#   Systems
# ----------------------------------------------------------------------
## @ingroup Methods-Weights-Correlations-General_Aviation
def systems(W_uav, V_fuel, V_int, N_tank, N_eng, l_fuselage, span, TOW, Nult, num_seats,  mach_number, has_air_conditioner=1):
    """ output = SUAVE.Methods.Weights.Correlations.General_Avation.systems(num_seats, ctrl_type, S_h, S_v, S_gross_w, ac_type)
        Calculate the weight of the different engine systems on the aircraft

        Source:
            Raymer, Aircraft Design: A Conceptual Approach (pg 461 in 4th edition)

        Inputs:
            V_fuel              - total fuel volume                     [meters**3]
            V_int               - internal fuel volume                  [meters**3]
            N_tank              - number of fuel tanks                  [dimensionless]
            N_eng               - number of engines                     [dimensionless]
            span                - wingspan                              [meters]
            TOW                 - gross takeoff weight of the aircraft  [kg]
            num_seats           - total number of seats on the aircraft [dimensionless]
            mach_number         - mach number                           [dimensionless]
            has_air_conditioner - integer of 1 if the vehicle has ac, 0 if not

        Outputs:
            output - a data dictionary with fields:
                wt_flt_ctrl - weight of the flight control system [kilograms]
                wt_apu - weight of the apu [kilograms]
                wt_hyd_pnu - weight of the hydraulics and pneumatics [kilograms]
                wt_avionics - weight of the avionics [kilograms]
                wt_opitems - weight of the optional items based on the type of aircraft [kilograms]
                wt_elec - weight of the electrical items [kilograms]
                wt_ac - weight of the air conditioning and anti-ice system [kilograms]
                wt_furnish - weight of the furnishings in the fuselage [kilograms]
    """ 
    # unpack inputs

    Q_tot  = V_fuel/Units.gallons
    Q_int  = V_int/Units.gallons

    l_fus  = l_fuselage / Units.ft  # Convert meters to ft
    b_wing = span/Units.ft

    W_0 = TOW/Units.lb
    
    #fuel system
    fuel_sys_wt = 2.49*(Q_tot**.726)*((Q_tot/(Q_tot+Q_int))**.363)*(N_tank**.242)*(N_eng**.157)*Units.lb

    #flight controls
    flt_ctrl_wt = .053*(l_fus**1.536)*(b_wing**.371)*((Nult*W_0**(10.**(-4.)))**.8)*Units.lb
    #Hydraulics & Pneumatics Group Wt
    hyd_pnu_wt = (.001*W_0) * Units.lb

    #avionics weight
    avionics_wt = 2.117*((W_uav/Units.lbs)**.933)*Units.lb 

    # Electrical Group Wt
    elec_wt = 12.57*((avionics_wt/Units.lb + fuel_sys_wt/Units.lb)**.51)*Units.lb

    # Environmental Control 
    ac_wt = has_air_conditioner*.265*(W_0**.52)*((1. * num_seats)**.68)*((avionics_wt/Units.lb)**.17)*(mach_number**.08)*Units.lb

    # Furnishings Group Wt
    furnish_wt = (.0582*W_0-65.)*Units.lb

    # packup outputs
    output = Data()   
    output.wt_flight_control    = flt_ctrl_wt
    output.wt_hyd_pnu           = hyd_pnu_wt
    output.wt_avionics          = avionics_wt
    output.wt_elec              = elec_wt
    output.wt_ac                = ac_wt
    output.wt_furnish           = furnish_wt
    output.wt_fuel_sys          = fuel_sys_wt
    output.wt_systems           = output.wt_flight_control + output.wt_hyd_pnu \
                                  + output.wt_ac + output.wt_avionics + output.wt_elec \
                                  + output.wt_furnish + output.wt_fuel_sys

    return output