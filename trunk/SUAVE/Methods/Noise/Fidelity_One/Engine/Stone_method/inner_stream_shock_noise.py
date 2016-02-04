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

def inner_stream_shock_noise ():
#Mi: inner stream Mach number fully expanded to ambient conditions
#Md_I: inner stream nozzle design Mach number
#D1iex: inner stream nozzle exit inner diamenter
#D2iex: inner stream nozzle exit outer diamenter
#Vi: fully expanded jet velocity of the inner stream
#Dhiex: hydraulic diameter of the inner stream nozzle exit

    Dhiex = D2iex - D1iex

    beta_i=((Mi**2-Md_I**2)**2+0.01*Md_I/(1-D1iex/D2iex))**0.25

    theta_m_i = 180 - np.asin(1/Mi)

    if (theta_cor_i < theta_m_i):
        F = 0.0
    elif (theta_cor_i >= theta_m_i):
        F = 0.75*(theta_cor_i - theta_m_i)

    Ci_sh = 158.0 - 50*np.log(1+(D1iex/D2iex)**2)

    Str_i_sh = (f*beta_i*Dhiexi/(0.7*Vi))*(1-Mf*np.cos(theta_cor_i))*((1+(0.7*Vi/rho_amb)*np.cos(theta_cor_i))**2 + \
        0.04*(0.7*Vi/sound_amb)**2)**0.5

    Xs_i_sh = (L1+2*Dhiex*(Mi**2-1)**0.5)/LSF

    OASPL_i_sh= Ci_sh + 10*np.log10((rho_amb/rho_isa)**2 * (sound_amb/sound_isa)**4)  + np.log10(Ai/Rcor_i**2) + \
        10*np.log10(beta_i**4/(1+beta_i**4)-40*np.log10(1-Mf*np.cos(theta_cor_i))-F)

    return()