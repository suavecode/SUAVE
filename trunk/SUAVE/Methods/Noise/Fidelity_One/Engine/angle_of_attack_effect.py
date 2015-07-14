#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      CARIDSIL
#
# Created:     08/07/2015
# Copyright:   (c) CARIDSIL 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np

def angle_of_attack_effect (AOA,Mach_aircraft,theta_m):
    """This function calculates the angle of attack effect, in decibels, to be added to the predicted mixed jet noise level."""
#AOA = angle of attack

    #Angle of attack effect
    ATK_m=0.5*AOA*Mach_aircraft*((1.8*theta_m/np.pi)-0.6)**2

    return(ATK_m)
