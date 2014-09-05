# compute_max_lift_coeff.py
#
# Created:  Anil V., Dec 2013
# Modified: Tarik, Feb 2014

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

#SUave Imports
import SUAVE
from SUAVE.Attributes import Units
from SUAVE.Components import Wings

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp
from SUAVE.Methods.Aerodynamics.Lift.High_lift_correlations.compute_slat_lift import compute_slat_lift
from SUAVE.Methods.Aerodynamics.Lift.High_lift_correlations.compute_flap_lift import compute_flap_lift

# ----------------------------------------------------------------------
#  compute_max_lift_coeff
# ----------------------------------------------------------------------
def compute_max_lift_coeff(vehicle,conditions=None):
    """ SUAVE.Methods.Aerodynamics.compute_max_lift_coeff(vehicle):
        Computes the maximum lift coefficient associated with an aircraft high lift system

        Inputs:
            vehicle - SUave type vehicle

            conditions - data dictionary with fields:
                mach_number - float or 1D array of freestream mach numbers
                airspeed    - float or 1D array of freestream airspeed
                rho         - air density
                mu          - air viscosity



            geometry - Not used


        Outputs:
            Cl_max_ls - maximum lift coefficient
            Cd_ind    - induced drag increment due to high lift device


        Assumptions:
            if needed

    """


    # initializing Cl and CDi
    Cl_max_ls = 0
    Cd_ind    = 0

    #unpack
    max_lift_coefficient_factor = vehicle.max_lift_coefficient_factor
    for wing in vehicle.wings:
    
        if not isinstance(wing,Wings.Main_Wing): continue
        #geometrical data
        Sref       = vehicle.S
        Swing      = wing.sref
        tc         = wing.t_c * 100
        chord_mac  = wing.chord_mac
        sweep      = wing.sweep
        taper      = wing.taper
        flap_chord = wing.flaps_chord
        flap_angle = wing.flaps_angle
        slat_angle = wing.slats_angle
        Swf        = wing.S_affected  #portion of wing area with flaps
        flap_type  = wing.flap_type

        # conditions data
        V    =  conditions.V
        roc  =  conditions.rho
        nu   =  conditions.mew

        ##Mcr  =  segment.M

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
        Cl_max_FAA= 1.1 * w_Clmax

        #-----------wing mounted engine ----
        Cl_max_w_eng= Cl_max_FAA - 0.2

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
    vehicle.S        = 92.        # m^2
    vehicle.max_lift_coefficient_factor = 1.10

    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'Main Wing'

    wing.sref      = vehicle.S       #
    wing.sweep     = 22. * Units.deg #
    wing.symmetric = True            #
    wing.t_c       = 0.11            #
    wing.taper     = 0.28            #
    wing.chord_mac   = 3.66

    wing.flaps_chord = 0.28
    wing.flaps_angle = 30.  * Units.deg
    wing.slats_angle = 15.  * Units.deg
    wing.S_affected  = 0.60 * wing.sref
    wing.flap_type   = 'double_slat'

    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Horizontal Stabilizer'

    wing.sref      = 26.         #
    #wing.span      = 100            #
    wing.sweep     = 34.5 * Units.deg #
    wing.symmetric = True
    wing.t_c       = 0.11          #
    wing.chord_mac   = 2.

    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #   Vertcal Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Vertcal Stabilizer'
    wing.sref      = 16.0        #
    wing.sweep     = 35. * Units.deg  #
    wing.symmetric = False
    wing.t_c       = 0.12          #
    wing.taper     = 0.10          #
    wing.chord_mac   = 2

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------

    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'Fuselage'

    fuselage.num_coach_seats = 114  #
    fuselage.seat_pitch      = 0.7455    # m
    fuselage.seats_abreast   = 4    #
    fuselage.fineness_nose   = 2.0  #
    fuselage.fineness_tail   = 3.0  #
    fuselage.fwdspace        = 0    #
    fuselage.aftspace        = 0    #
    fuselage.width           = 3.0  #
    fuselage.height          = 3.4  #

    # add to vehicle
    vehicle.append_component(fuselage)

    condition = SUAVE.Structure.Data()
    condition.M = 0.3
    condition.V = 51. #m/s
    condition.rho = 1.1225 #kg/m?
    condition.mew = 1.79E-05

    Cl_max_ls, Cd_ind = compute_max_lift_coeff(vehicle,condition)
    print 'CLmax : ', Cl_max_ls, 'dCDi :' , Cd_ind

