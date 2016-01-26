# plug_downstream_shock_noise.py
#
# Created:  Carlos, August 2015
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

def plug_downstream_shock_noise ():
#Am: fully expanded merged area
#Md: isentropically-calculated Mach number based the mass-average specif heat ratio and the fully expanded are-weight pressure ratio
#Md_D: overall nozzle design Mach number
#Dhdex: total hydraulic nozzle exit diameter
#Dm: equivalent diameter based on total fan plus core fully-expanded merged area

    Am=Ai+Ao
    Pd=(Pi*Ai+Po*Ao)/(Ao+Ai)
    Md=(((Pd/P_amb)**((gama-1)/gama)-1)*(2/(gama-1)))**0.5
    Md_D=(1-0.5*(1-2*rpt/D2oex))*(Md_i+Md_o*BPR)/(1+BPR)+0.5*(1-2*rpt/D2oex)*Md
    beta_d=((Md**2-Md_D**2)**2+0.01*Md_D/(1-D1iex/D2oex))**0.25

    theta_m_d=180-np.asin(1/Md)

    if (theta_cor_d < theta_m_d):
        F = 0.0
    elif (theta_cor_d >= theta_m_d):
        F = 0.75*(theta_cor_d - theta_m_d)

    Cd_sh= 164.5 -50*np.log10(1+(Diiex/D2oex)**2)

    Dhdex=D2oex-Diiex

    V_d = Md_D*(gama*R_gas*t_d)**0.5
    Td=(mflow_i*Ti+mflow_o*To)/(mflow_i+mflow_o)
    t_d = Td/(1+Md**2*(gama-1)/2)

    Dm=(4*(Ai+Ao)/np.pi)**0.5
    Xs_d_sh = (Lp + L1 + 2*Dm*(Md**2-1)**0.5)/LSF

    Str_d_sh = (f*beta_d*Dhdex/(0.7*V_d))*(1-Mf*np.cos(theta_cor_d))*((1+(0.7*V_d/sound_amb)*np.cos(theta_cor_d))**2 + \
        0.04*(0.7*V_d/sound_amb)**2)**0.5

    OASPL_d_sh= Cd_sh + 10*np.log10((rho_amb/rho_isa)**2 * (sound_amb/sound_isa)**4)  + np.log10(Am/Rcor_d**2) + \
        10*np.log10(beta_d**4/(1+beta_d**4)-40*np.log10(1-Mf*np.cos(theta_cor_d))-F)

    return()