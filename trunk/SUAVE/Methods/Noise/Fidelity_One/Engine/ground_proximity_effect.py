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

def ground_proximity_effect (Velocity_mixed,sound_ambient,theta_m,engine_height,Diameter_mixed,frequency):
    """This function calculates the ground proximity effect, in decibels, and is used for full-scale engine test stand."""

    #Ground proximity is applied only for the mixed jet component
    GPROX_m=(5*Velocity_mixed/sound_ambient)*np.exp(-(9*(theta_m/np.pi)-6.75)**2- \
        ((engine_height/Diameter_mixed)-2.5)**2)*(1+(np.sin((np.pi*engine_height*frequency/sound_ambient)-np.pi/2))**2)/ \
        (2+np.abs((engine_height*frequency/sound_ambient)-1))

    return(GPROX_m)