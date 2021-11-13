## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
# compute_max_lift_coeff.py
#
# Created:  Dec 2013, A. Variyar
# Modified: Feb 2014, T. Orra
#           Jan 2016, E. Botero        
#           Feb 2019, E. Botero      
#           Jul 2020, E. Botero 
#           May 2021, E. Botero  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

#SUAVE Imports
import SUAVE
from SUAVE.Core import Units
from SUAVE.Components import Wings
from SUAVE.Core  import Data

from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift.compute_slat_lift import compute_slat_lift
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift.compute_flap_lift import compute_flap_lift

# ----------------------------------------------------------------------
#  compute_max_lift_coeff
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
def compute_max_lift_coeff(state,settings,geometry):
    """Computes the maximum lift coefficient associated with an aircraft high lift system

    Assumptions:
    None

    Source:
    Unknown

    Inputs:
    analyses.max_lift_coefficient_factor       [Unitless]
    vehicle.reference_area                     [m^2]
    vehicle.wings.                             
      areas.reference                          [m^2]
      thickness_to_chord                       [Unitless]
      chords.mean_aerodynamic                  [m]
      sweeps.quarter_chord                     [radians]
      taper                                    [Unitless]
      flaps.chord                              [m]
     control_surfaces.flap.deflection          [radians]
     control_surfaces.slat.deflection          [radians]
      areas.affected                           [m^2]
      control_surfaces.flap.configuration_type [string]
    conditions.freestream.                     
      velocity                                 [m/s]
      density                                  [kg/m^3]
      dynamic_viscosity                        [N s/m^2]
                                               
    Outputs:                                   
    Cl_max_ls (maximum CL)                     [Unitless]
    Cd_ind    (induced drag)                   [Unitless]

    Properties Used:
    N/A
    """    


    # initializing Cl and CDi
    Cl_max_ls = 0
    Cd_ind    = 0
    vehicle = geometry
    conditions = state.conditions

    #unpack
    max_lift_coefficient_factor = settings.maximum_lift_coefficient_factor
    for wing in vehicle.wings:
    
        if not wing.high_lift: continue
        #geometrical data
        Sref       = vehicle.reference_area
        Swing      = wing.areas.reference
        tc         = wing.thickness_to_chord * 100
        chord_mac  = wing.chords.mean_aerodynamic
        sweep      = wing.sweeps.quarter_chord
        sweep_deg  = wing.sweeps.quarter_chord / Units.degree # convert into degrees
        taper      = wing.taper
        
        # conditions data
        V    = conditions.freestream.velocity
        roc  = conditions.freestream.density
        nu   = conditions.freestream.dynamic_viscosity

        #--cl max based on airfoil t_c
        Cl_max_ref = -0.0009*tc**3 + 0.0217*tc**2 - 0.0442*tc + 0.7005
        #-reynolds number effect
        Reyn     =  V * roc * chord_mac / nu
        Re_ref   = 9*10**6
        op_Clmax = Cl_max_ref * ( Reyn / Re_ref ) **0.1

        #wing cl_max to outer panel Cl_max
        w_Clmax = op_Clmax* ( 0.919729714285715 -0.044504761904771*taper \
                             -0.001835900000000*sweep_deg +  0.247071428571446*taper**2 +  \
                              0.003191500000000*taper*sweep_deg -0.000056632142857*sweep_deg**2  \
                             -0.279166666666676*taper**3 +  0.002300000000000*taper**2*sweep_deg + \
                              0.000049982142857*taper*sweep_deg**2  -0.000000280000000* sweep_deg**3)

        #---FAR stall speed effect---------------
        #should be optional based on aircraft being modelled
        Cl_max_FAA = 1.1 * w_Clmax

        #-----------wing mounted engine ----
        Cl_max_w_eng = Cl_max_FAA - 0.2

        # Compute CL increment due to Flap
        if 'slat' in wing.control_surfaces.keys():
         # Compute CL increment due to Slat
            slat_angle = wing.control_surfaces.slat.deflection
            dcl_slat = compute_slat_lift(slat_angle, sweep)
        else:
            dcl_slat = 0.

        # Compute CL increment due to Flap
        if 'flap' in wing.control_surfaces.keys():
            flap_type  = wing.control_surfaces.flap.configuration_type
            flap_chord = wing.control_surfaces.flap.chord_fraction # correct !!! 
            flap_angle = wing.control_surfaces.flap.deflection
            Swf        = wing.areas.affected  # portion of wing area with flaps
            dcl_flap   = compute_flap_lift(tc,flap_type,flap_chord,flap_angle,sweep,Sref,Swf)
        else:
            dcl_flap = 0.0

        #results
        Cl_max_ls += (Cl_max_w_eng + dcl_slat + dcl_flap) * Swing / Sref
        Cd_ind += ( 0.01 ) * Swing / Sref

    Cl_max_ls = Cl_max_ls * max_lift_coefficient_factor
    return Cl_max_ls, Cd_ind
