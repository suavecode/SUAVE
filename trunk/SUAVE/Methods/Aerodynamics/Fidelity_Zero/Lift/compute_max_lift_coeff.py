## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
# compute_max_lift_coeff.py
#
# Created:  Dec 2013, A. Variyar
# Modified: Feb 2014, T. Orra
#           Jan 2016, E. Botero         

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
def compute_max_lift_coeff(vehicle,conditions=None):
    """Computes the maximum lift coefficient associated with an aircraft high lift system

    Assumptions:
    None

    Source:
    Unknown

    Inputs:
    vehicle.max_lift_coefficient_factor [Unitless]
    vehicle.reference_area              [m^2]
    vehicle.wings. 
      areas.reference                   [m^2]
      thickness_to_chord                [Unitless]
      chords.mean_aerodynamic           [m]
      sweeps.quarter_chord              [radians]
      taper                             [Unitless]
      flaps.chord                       [m]
      flaps.angle                       [radians]
      slats.angle                       [radians]
      areas.affected                    [m^2]
      flaps.type                        [string]
    conditions.freestream.
      velocity                          [m/s]
      density                           [kg/m^3]
      dynamic_viscosity                 [N s/m^2]

    Outputs:
    Cl_max_ls (maximum CL)              [Unitless]
    Cd_ind    (induced drag)            [Unitless]

    Properties Used:
    N/A
    """    


    # initializing Cl and CDi
    Cl_max_ls = 0
    Cd_ind    = 0

    #unpack
    max_lift_coefficient_factor = vehicle.max_lift_coefficient_factor
    for wing in vehicle.wings:
    
        if not wing.high_lift: continue
        #geometrical data
        Sref       = vehicle.reference_area
        Swing      = wing.areas.reference
        tc         = wing.thickness_to_chord * 100
        chord_mac  = wing.chords.mean_aerodynamic
        sweep      = wing.sweeps.quarter_chord # convert into degrees
        taper      = wing.taper
        flap_chord = wing.flaps.chord
        flap_angle = wing.flaps.angle
        slat_angle = wing.slats.angle
        Swf        = wing.areas.affected  #portion of wing area with flaps
        flap_type  = wing.flaps.type
        
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
                             -0.001835900000000*sweep +  0.247071428571446*taper**2 +  \
                              0.003191500000000*taper*sweep  -0.000056632142857*sweep**2  \
                             -0.279166666666676*taper**3 +  0.002300000000000*taper**2*sweep + \
                              0.000049982142857*taper*sweep**2  -0.000000280000000* sweep**3)

        #---FAR stall speed effect---------------
        #should be optional based on aircraft being modelled
        Cl_max_FAA = 1.1 * w_Clmax

        #-----------wing mounted engine ----
        Cl_max_w_eng = Cl_max_FAA - 0.2

        # Compute CL increment due to Slat
        dcl_slat = compute_slat_lift(slat_angle, sweep)

         # Compute CL increment due to Flap
        dcl_flap = compute_flap_lift(tc,flap_type,flap_chord,flap_angle,sweep,Sref,Swf)

        #results
        Cl_max_ls += (Cl_max_w_eng + dcl_slat + dcl_flap) * Swing / Sref
        Cd_ind += ( 0.01 ) * Swing / Sref

    Cl_max_ls = Cl_max_ls * max_lift_coefficient_factor
    return Cl_max_ls, Cd_ind


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------
    vehicle = SUAVE.Vehicle()
    # basic data
    vehicle.reference_area              = 92.        # m^2
    vehicle.max_lift_coefficient_factor = 1.10

    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'Main Wing'

    wing.areas.reference         = vehicle.reference_area
    wing.sweeps.quarter_chord    = 22. * Units.deg
    wing.symmetric               = True
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.28
    wing.chords.mean_aerodynamic = 3.66

    wing.flaps.chord = 0.28
    wing.flaps.angle = 30.  * Units.deg
    wing.slats.angle = 15.  * Units.deg
    wing.areas.affected  = 0.60 * wing.areas.reference 
    wing.flaps.type   = 'double_slat'
    
    wing.high_lift  = True

    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Horizontal Stabilizer'

    wing.areas.reference         = 26.
    wing.sweeps.quarter_chord    = 34.5 * Units.deg
    wing.symmetric               = True
    wing.thickness_to_chord      = 0.11
    wing.chords.mean_aerodynamic = 2.

    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Vertical Stabilizer'
    wing.areas.reference         = 16.0
    wing.sweeps.quarter_chord    = 35. * Units.deg
    wing.symmetric               = False
    wing.thickness_to_chord      = 0.12
    wing.taper                   = 0.10
    wing.chords.mean_aerodynamic = 2

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------

    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'Fuselage'

    fuselage.number_coach_seats = 114  #
    fuselage.seat_pitch         = 0.7455    # m
    fuselage.seats_abreast      = 4    #
    fuselage.fineness.nose      = 2.0  #
    fuselage.fineness.tail      = 3.0  #
    fuselage.fwdspace           = 0    #
    fuselage.aftspace           = 0    #
    fuselage.width              = 3.0  #
    fuselage.heights.maximum    = 3.4  #

    # add to vehicle
    vehicle.append_component(fuselage)

    conditions = Data()
    conditions.freestream = Data()
    conditions.freestream.mach_number = 0.3
    conditions.freestream.velocity    = 51. #m/s
    conditions.freestream.density     = 1.1225 #kg/m?
    conditions.freestream.dynamic_viscosity   = 1.79E-05


    Cl_max_ls, Cd_ind = compute_max_lift_coeff(vehicle,conditions)
    print('CLmax : ', Cl_max_ls, 'dCDi :' , Cd_ind)

