# propeller_design.py
# 
# Created:  Jul 2014, E. Botero
# Modified: Feb 2016, E. Botero
#           Jul 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
import numpy as np
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Propeller Design
# ----------------------------------------------------------------------
    
def propeller_design(prop_attributes):
    """ Optimizes propeller chord and twist given input parameters.
          
          Inputs:
              Either design power or thrust
              hub radius
              tip radius
              rotation rate
              freestream velocity
              number of blades
              number of stations
              design lift coefficient
              airfoil data

          Outputs:
              Twist distribution
              Chord distribution
              
          Assumptions:
              Based on Design of Optimum Propellers by Adkins and Liebeck

    """    
    # Unpack
    B      = prop_attributes.number_blades
    R      = prop_attributes.tip_radius
    Rh     = prop_attributes.hub_radius
    omega  = prop_attributes.angular_velocity    # Rotation Rate in rad/s
    V      = prop_attributes.freestream_velocity # Freestream Velocity
    Cl     = prop_attributes.design_Cl           # Design Lift Coefficient
    alt    = prop_attributes.design_altitude
    Thrust = prop_attributes.design_thrust
    Power  = prop_attributes.design_power
    
    # Calculate atmospheric properties
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data = atmosphere.compute_values(alt)
    
    p   = atmo_data.pressure[0]
    T   = atmo_data.temperature[0]
    rho = atmo_data.density[0]
    a   = atmo_data.speed_of_sound[0]
    mu  = atmo_data.dynamic_viscosity[0]
    nu  = mu/rho
    
    # Nondimensional thrust
    Tc = 2.*Thrust/(rho*(V*V)*np.pi*(R*R))
    Pc = 2.*Power/(rho*(V*V*V)*np.pi*(R*R))    
    
    tol   = 1e-10 # Convergence tolerance
    N     = 20   # Number of Stations

    #Step 1, assume a zeta
    zeta = 0.1 # Assume to be small initially
    
    #Step 2, determine F and phi at each blade station
    
    chi0    = Rh/R # Where the propeller blade actually starts
    chi     = np.linspace(chi0,1,N+1) # Vector of nondimensional radii
    chi     = chi[0:N]
    lamda   = V/(omega*R)             # Speed ratio
    r       = chi*R                   # Radial coordinate
    x       = omega*r/V               # Nondimensional distance
    diff    = 1.0                     # Difference between zetas
    n       = omega/(2*np.pi)         # Cycles per second
    D       = 2.*R
    J       = V/(D*n)
    
    while diff>tol:
        #Things that need a loop
        Tcnew   = Tc
        tanphit = lamda*(1.+zeta/2.)   # Tangent of the flow angle at the tip
        phit    = np.arctan(tanphit)   # Flow angle at the tip
        tanphi  = tanphit/chi          # Flow angle at every station
        f       = (B/2.)*(1.-chi)/np.sin(phit) 
        F       = (2./np.pi)*np.arccos(np.exp(-f)) #Prandtl momentum loss factor
        phi     = np.arctan(tanphi)  #Flow angle at every station
        
        #Step 3, determine the product Wc, and RE
        G       = F*x*np.cos(phi)*np.sin(phi) #Circulation function
        Wc      = 4.*np.pi*lamda*G*V*R*zeta/(Cl*B)
        Ma      = Wc/a
        RE      = Wc/nu

        #Step 4, determine epsilon and alpha from airfoil data
        
        #This is an atrocious fit of DAE51 data at RE=50k for Cd
        #There is also RE scaling
        Cdval   = (0.108*(Cl**4)-0.2612*(Cl**3)+0.181*(Cl**2)-0.0139*Cl+0.0278)*((50000./RE)**0.2)

        #More Cd scaling from Mach from AA241ab notes for turbulent skin friction
        Tw_Tinf = 1. + 1.78*(Ma**2)
        Tp_Tinf = 1. + 0.035*(Ma**2) + 0.45*(Tw_Tinf-1.)
        Tp      = Tp_Tinf*T
        Rp_Rinf = (Tp_Tinf**2.5)*(Tp+110.4)/(T+110.4)
        
        Cd      = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval
        
        alpha   = Cl/(2.*np.pi)
        epsilon = Cd/Cl
        
        #Step 5, change Cl and repeat steps 3 and 4 until epsilon is minimized
        
        #Step 6, determine a and a', and W
        
        a       = (zeta/2.)*(np.cos(phi)**2.)*(1.-epsilon*np.tan(phi))
        aprime  = (zeta/(2.*x))*np.cos(phi)*np.sin(phi)*(1.+epsilon/np.tan(phi))
        W       = V*(1.+a)/np.sin(phi)
        
        #Step 7, compute the chord length and blade twist angle    
        
        c       = Wc/W
        beta    = alpha + phi # Blade twist angle
    
        #Step 8, determine 4 derivatives in I and J
    
        Iprime1 = 4.*chi*G*(1.-epsilon*np.tan(phi))
        Iprime2 = lamda*(Iprime1/(2.*chi))*(1.+epsilon/np.tan(phi)
                                            )*np.sin(phi)*np.cos(phi)
        Jprime1 = 4.*chi*G*(1.+epsilon/np.tan(phi))
        Jprime2 = (Jprime1/2.)*(1.-epsilon*np.tan(phi))*(np.cos(phi)**2.)
        
        dR      = (r[1]-r[0])*np.ones_like(Jprime1)
        dchi    = (chi[1]-chi[0])*np.ones_like(Jprime1)
        
        #Integrate derivatives from chi=chi0 to chi=1
        
        I1      = np.dot(Iprime1,dchi)
        I2      = np.dot(Iprime2,dchi)
        J1      = np.dot(Jprime1,dchi)
        J2      = np.dot(Jprime2,dchi)        

        #Step 9, determine zeta and and Pc or zeta and Tc
        
        if (Pc==0.)&(Tc!=0.): 
            #First Case, Thrust is given
            #Check to see if Tc is feasible, otherwise try a reasonable number
            if Tcnew>=I2*(I1/(2.*I2))**2.:
                Tcnew = I2*(I1/(2.*I2))**2.
            zetan    = (I1/(2.*I2)) - ((I1/(2.*I2))**2.-Tcnew/I2)**0.5

        elif (Pc!=0.)&(Tc==0.): 
            #Second Case, Thrust is given
            zetan    = -(J1/(J2*2.)) + ((J1/(J2*2.))**2.+Pc/J2)**0.5
            
        else:
            print('Power and thrust are both specified!')
    
        #Step 10, repeat starting at step 2 with the new zeta
        diff = abs(zeta-zetan)
        
        zeta = zetan
    
    #Step 11, determine propeller efficiency etc...
    
    if (Pc==0.)&(Tc!=0.): 
        if Tcnew>=I2*(I1/(2.*I2))**2.:
            Tcnew = I2*(I1/(2.*I2))**2.
            print('Tc infeasible, reset to:')
            print(Tcnew)        
        #First Case, Thrust is given
        zeta    = (I1/(2.*I2)) - ((I1/(2.*I2))**2.-Tcnew/I2)**0.5
        Pc      = J1*zeta + J2*(zeta**2.)
        Tc      = I1*zeta - I2*(zeta**2.)
        
    elif (Pc!=0.)&(Tc==0.): 
        #Second Case, Thrust is given
        zeta    = -(J1/(2.*J2)) + ((J1/(2.*J2))**2.+Pc/J2)**0.5
        Tc      = I1*zeta - I2*(zeta**2.)
        Pc      = J1*zeta + J2*(zeta**2.)
        
    else:
        print('Power and thrust are both specified!')    
        
    # Calculate mid-chord alignment angle, MCA
    # This is the distance from the mid chord to the line axis out of the center of the blade
    # In this case the 1/4 chords are all aligned
    
    MCA = c/4. - c[0]/4.
    
    
    Power = Pc*rho*(V**3)*np.pi*(R**2)/2
    Cp    = Power/(rho*(n**3)*(D**5))

    prop_attributes.twist_distribution = beta
    prop_attributes.chord_distribution = c
    prop_attributes.Cp                 = Cp
    prop_attributes.mid_chord_aligment = MCA
    
    #These are used to check, the values here were used to verify against
    #AIAA 89-2048 for their propeller
    
    return prop_attributes