# outer_stream_shock_noise.py
#
# Created:  Carlos, August 2015
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

def outer_stream_shock_noise ():
#Ao: fully expanded area of the outer stream
#Mo: outer stream Mach number fully expanded to ambient conditions
#Md_O: outer stream nozzle design Mach number
#Dhoex: hydraulic diameter of the outer stream nozzle exit
#Deqoex: equivalent diameter of the outer stream nozzle exit

    Dhoex = D2oex - D1oex
    Deqoex = (D2oex**2 - D1oex**2)**0.5

    beta_o=((Mo**2-Md_O**2)**2+0.01*Md_O/(1-D1oex/D2oex))**0.25

    theta_m_o=180-np.asin(1/Mo)

    if (theta_cor_o < theta_m_o):
        F = 0.0
    elif (theta_cor_o >= theta_m_o):
        F = 0.75*(theta_cor_o - theta_m_o)
    if Mi>=1.0:
        Co_sh = 168.0 - 60*np.log10(1+(D1oex/D2oex)**2)
    elif Mi<1.0:
        Co_sh = 163.5 - 45*np.log10(1+(D1oex/D2oex)**2)

    Str_o_sh=(f*beta_o*(Dhoex**0.9*Deqoex**0.1)/(0.7*Vo))*(1-Mf*np.cos(theta_cor_o))*((1+(0.7*Vo/rho_amb)*np.cos(theta_cor_o))**2 + \
        0.04*(0.7*Vo/sound_amb)**2)**0.5

    Xs_o_sh = (2*Dhoex*(Mo**2-1)**0.5)/LSF

    OASPL_o_sh= Co_sh + 10*np.log10((rho_amb/rho_isa)**2 * (sound_amb/sound_isa)**4)  + np.log10(Ao/Rcor_o**2) + \
        10*np.log10(beta_o**4/(1+beta_o**4)-40*np.log10(1-Mf*np.cos(theta_cor_o))-F)

    return()