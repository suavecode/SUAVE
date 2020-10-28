## @ingroup Methods-Propulsion
# propeller_design.py
# 
# Created:  Jul 2014, E. Botero
# Modified: Feb 2016, E. Botero
#           Jul 2017, M. Clarke
#           Mar 2020, M. Clarke
#           Sep 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
import numpy as np
import scipy as sp
from SUAVE.Core import Units , Data
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry \
     import import_airfoil_geometry

from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_polars \
     import compute_airfoil_polars
# ----------------------------------------------------------------------
#  Propeller Design
# ----------------------------------------------------------------------

def propeller_design(prop,number_of_stations=20):
    """ Optimizes propeller chord and twist given input parameters.
          
          Inputs:
          Either design power or thrust
          prop_attributes.
            hub radius                       [m]
            tip radius                       [m]
            rotation rate                    [rad/s]
            freestream velocity              [m/s]
            number of blades               
            number of stations
            design lift coefficient
            airfoil data
            
          Outputs:
          Twist distribution                 [array of radians]
          Chord distribution                 [array of meters]
              
          Assumptions/ Source:
          Based on Design of Optimum Propellers by Adkins and Liebeck
          
    """    
    # Unpack
    N      = number_of_stations       # this number determines the discretization of the propeller into stations 
    B      = prop.number_blades
    R      = prop.tip_radius
    Rh     = prop.hub_radius
    omega  = prop.angular_velocity    # Rotation Rate in rad/s
    V      = prop.freestream_velocity # Freestream Velocity
    Cl     = prop.design_Cl           # Design Lift Coefficient
    alt    = prop.design_altitude
    Thrust = prop.design_thrust
    Power  = prop.design_power
    a_geo  = prop.airfoil_geometry
    a_pol  = prop.airfoil_polars        
    a_loc  = prop.airfoil_polar_stations    
    
    if (Thrust == None) and (Power== None):
        raise AssertionError('Specify either design thrust or design power!')
    
    elif (Thrust!= None) and (Power!= None):
        raise AssertionError('Specify either design thrust or design power!')
    
    if prop.rotation == None:
        prop.rotation = list(np.ones(int(B))) 
        
    # Calculate atmospheric properties
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data = atmosphere.compute_values(alt)
    
    p              = atmo_data.pressure[0]
    T              = atmo_data.temperature[0]
    rho            = atmo_data.density[0]
    speed_of_sound = atmo_data.speed_of_sound[0]
    mu             = atmo_data.dynamic_viscosity[0]
    nu             = mu/rho
    
    # Nondimensional thrust
    if (Thrust!= None) and (Power == None):
        Tc = 2.*Thrust/(rho*(V*V)*np.pi*(R*R))     
        Pc = 0.0 
    
    elif (Thrust== None) and (Power != None):
        Tc = 0.0   
        Pc = 2.*Power/(rho*(V*V*V)*np.pi*(R*R))  
    
    tol   = 1e-10 # Convergence tolerance

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
    
    c = 0.2 * np.ones_like(chi)
    
    # if user defines airfoil, check dimension of stations
    if  a_pol != None and a_loc != None:
        if len(a_loc) != N:
            raise AssertionError('\nDimension of airfoil sections must be equal to number of stations on propeller')
    
    else:
        # Import Airfoil from regression
        print('\nNo airfoils specified for propeller or rotor airfoil specified. \nDefaulting to NACA 4412 airfoils that will provide conservative estimates.') 
        import os
        ospath = os.path.abspath(__file__)
        path   = ospath.replace('\\','/').split('trunk/SUAVE/Methods/Propulsion/propeller_design.py')[0] \
            + 'regression/scripts/Vehicles/' 
        a_geo  = [ path +  'NACA_4412.txt'] 
        a_pol  = [[path +  'NACA_4412_polar_Re_50000.txt' ,
                   path +  'NACA_4412_polar_Re_100000.txt' ,
                   path +  'NACA_4412_polar_Re_200000.txt' ,
                   path +  'NACA_4412_polar_Re_500000.txt' ,
                   path +  'NACA_4412_polar_Re_1000000.txt' ]]   # airfoil polars for at different reynolds numbers  
       
        # 0 represents the first airfoil, 1 represents the second airfoil etc. 
        a_loc = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  
    
        prop.airfoil_geometry        = a_geo
        prop.airfoil_polars          = a_pol     
        prop.airfoil_polar_stations  = a_loc     
     
    while diff>tol:      
        # assign chord distribution
        prop.chord_distribution = c 
                         
        # compute airfoil polars for airfoils 
        airfoil_polars  = compute_airfoil_polars(prop, a_geo, a_pol)  
        airfoil_cl_surs = airfoil_polars.lift_coefficient_surrogates 
        airfoil_cd_surs = airfoil_polars.drag_coefficient_surrogates 
                         
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
        Ma      = Wc/speed_of_sound
        RE      = Wc/nu

        #Step 4, determine epsilon and alpha from airfoil data  
        alpha = np.zeros_like(RE)
        Cdval = np.zeros_like(RE)  
        for i in range(N):
            AoA_old_guess = 0.01
            cl_diff       = 1  
            broke         = False   
            ii            = 0
            
            # Newton Raphson Iteration 
            while cl_diff > 1E-3:
                
                Cl_guess       = airfoil_cl_surs[a_geo[a_loc[i]]](RE[i],AoA_old_guess,grid=False) - Cl 
                
                # central difference derivative 
                dx             = 1E-5
                dCL            = (airfoil_cl_surs[a_geo[a_loc[i]]](RE[i],AoA_old_guess + dx,grid=False) - airfoil_cl_surs[a_geo[a_loc[i]]](RE[i],AoA_old_guess- dx,grid=False))/ (2*dx)
                 
                # update AoA guess 
                AoA_new_guess  = AoA_old_guess - Cl_guess/dCL
                AoA_old_guess  = AoA_new_guess 
                
                # compute difference for tolerance check
                cl_diff        = abs(Cl_guess)      
                 
                ii+=1 	
                if ii>10000:	
                    # maximum iterations is 10000
                    print('Propeller/Rotor section is not converging to solution')
                    broke = True	
                    break                    
                
            alpha[i] = AoA_old_guess     
            Cdval[i] = airfoil_cd_surs[a_geo[a_loc[i]]](RE[i],alpha[i],grid=False)  

        #More Cd scaling from Mach from AA241ab notes for turbulent skin friction
        Tw_Tinf = 1. + 1.78*(Ma**2)
        Tp_Tinf = 1. + 0.035*(Ma**2) + 0.45*(Tw_Tinf-1.)
        Tp      = Tp_Tinf*T
        Rp_Rinf = (Tp_Tinf**2.5)*(Tp+110.4)/(T+110.4) 
        Cd      = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval
        
        #Step 5, change Cl and repeat steps 3 and 4 until epsilon is minimized 
        epsilon = Cd/Cl  
        
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
        
    # Calculate mid-chord alignment angle, MCA
    # This is the distance from the mid chord to the line axis out of the center of the blade
    # In this case the 1/4 chords are all aligned 
    MCA    = c/4. - c[0]/4.
    
    Thrust = Tc*rho*(V**2)*np.pi*(R**2)/2
    Power  = Pc*rho*(V**3)*np.pi*(R**2)/2 
    Ct     = Thrust/(rho*(n*n)*(D*D*D*D))
    Cp     = Power/(rho*(n*n*n)*(D*D*D*D*D))  
    
    # compute max thickness distribution  
    t_max  = np.zeros(N)    
    t_c    = np.zeros(N)    
    airfoil_geometry_data = import_airfoil_geometry(a_geo)
    for i in range(N):
        t_c[i]   = airfoil_geometry_data.thickness_to_chord[a_loc[i]]    
        t_max[i] = airfoil_geometry_data.max_thickness[a_loc[i]]*c[i]
        
    # Nondimensional thrust
    if prop.design_power == None: 
        prop.design_power = Power[0]        
    elif prop.design_thrust == None: 
        prop.design_thrust = Thrust[0]      
    
    # approximate thickness to chord ratio  
    t_c_at_70_percent = t_c[int(N*0.7)]
    
    # blade solidity
    r          = chi*R                    # Radial coordinate   
    blade_area = sp.integrate.cumtrapz(B*c, r-r[0])
    sigma      = blade_area[-1]/(np.pi*R**2)   
    
    prop.design_torque              = Power[0]/omega
    prop.max_thickness_distribution = t_max
    prop.twist_distribution         = beta
    prop.chord_distribution         = c
    prop.radius_distribution        = r 
    prop.number_blades              = int(B)
    prop.design_power_coefficient   = Cp 
    prop.design_thrust_coefficient  = Ct 
    prop.mid_chord_aligment         = MCA
    prop.thickness_to_chord         = t_c_at_70_percent
    prop.blade_solidity             = sigma  
    prop.airfoil_cl_surrogates      = airfoil_cl_surs
    prop.airfoil_cd_surrogates      = airfoil_cd_surs 

    return prop
