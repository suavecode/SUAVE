#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:     
#
# Author:      CARIDSIL
#
# Created:     11/08/2015
# Copyright:   (c) CARIDSIL 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

def shock_noise_components():
    beta=((M**2-Md**2)**2+0.01*Md/(1-D1/D2))**0.25
    theta_M = 180 - np.degrees.asin(1/M)

    if (theta_cor < theta_M):
        F = 0.0
    elif (theta_cor >= theta_M):
        F = 0.75*(theta_cor - theta_M)

    Str_sh = (f*beta*Dh/(0.7*V))*(1-Mf*np.cos(theta_cor))*((1+(0.7*V/sound_amb)*np.cos(theta_cor))**2 + \
        alfa**2*(0.7*V/sound_amb)**2)**0.5

    Xs_sh=(L+2*Dh*(M**2-1)**0.5)/LSF

    OASPL_sh = C_sh + 10*np.log10((rho_amb/rho_isa)**2 * (sound_amb/sound_isa)**4)  + np.log10(A/Rcor**2) + \
        10*np.log10(beta**4/(1+beta**4)-40*np.log10(1-Mf*np.cos(theta_cor))-F)

    return()