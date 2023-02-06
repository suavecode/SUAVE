## @ingroup Methods-Propulsion
# propeller_design.py
# 
# Created:  Jul 2014, E. Botero
# Modified: Feb 2016, E. Botero
#           Jul 2017, M. Clarke
#           Mar 2020, M. Clarke
#           Sep 2020, M. Clarke
#           Feb 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import MARC
from MARC.Core import Data
from MARC.Core.Utilities import interp2d
import numpy as np
import scipy as sp 
from scipy.optimize import root 
from MARC.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_properties \
     import compute_airfoil_properties
from MARC.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series \
     import compute_naca_4series
from MARC.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry\
     import import_airfoil_geometry
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
            design_altitude                  [m]
            airfoil data
            
          Outputs:
          Twist distribution                 [array of radians]
          Chord distribution                 [array of meters]
              
          Assumptions/ Source:
          Based on Design of Optimum Propellers by Adkins and Liebeck
          
    """
    # Unpack
    N            = number_of_stations       # this number determines the discretization of the propeller into stations
    B            = prop.number_of_blades
    R            = prop.tip_radius
    Rh           = prop.hub_radius
    omega        = prop.cruise.design_angular_velocity    # Rotation Rate in rad/s
    V            = prop.cruise.design_freestream_velocity # Freestream Velocity
    Cl           = prop.cruise.design_Cl                  # Design Lift Coefficient
    alt          = prop.cruise.design_altitude
    Thrust       = prop.cruise.design_thrust
    Power        = prop.cruise.design_power
    airfoils     = prop.Airfoils 
    a_loc        = prop.airfoil_polar_stations
    
    if (Thrust == None) and (Power== None):
        raise AssertionError('Specify either design thrust or design power!')
    
    elif (Thrust!= None) and (Power!= None):
        raise AssertionError('Specify either design thrust or design power!') 
    
    if V == 0.0:
        V = 1E-6 
        
    # Calculate atmospheric properties
    atmosphere = MARC.Analyses.Atmospheric.US_Standard_1976()
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

    # Step 1, assume a zeta
    zeta = 0.1 # Assume to be small initially
    
    # Step 2, determine F and phi at each blade station
    
    chi0    = Rh/R # Where the propeller blade actually starts
    chi     = np.linspace(chi0,1,N+1) # Vector of nondimensional radii
    chi     = chi[0:N]
    lamda   = V/(omega*R)             # Speed ratio
    r       = chi*R                   # Radial coordinate
    x       = omega*r/V               # Nondimensional distance
    diff    = 1.0                     # Difference between zetas
    n       = omega/(2*np.pi)         # Cycles per second
    D       = 2.*R  
    c       = 0.2 * np.ones_like(chi)
    
    # if user defines airfoil, check dimension of stations
    num_airfoils = len(airfoils.keys())
    if num_airfoils>0:
        if len(a_loc) != N:
            raise AssertionError('\nDimension of airfoil sections must be equal to number of stations on propeller') 
        
        for _,airfoil in enumerate(airfoils):  
            if airfoil.geometry == None: # first, if airfoil geometry data not defined, import from geoemtry files
                if airfoil.NACA_4_series_flag: # check if naca 4 series of airfoil from datafile
                    airfoil.geometry = compute_naca_4series(airfoil.coordinate_file,airfoil.number_of_points)
                else:
                    airfoil.geometry = import_airfoil_geometry(airfoil.coordinate_file,airfoil.number_of_points) 
    
            if airfoil.polars == None: # compute airfoil polars for airfoils
                airfoil.polars = compute_airfoil_properties(airfoil.geometry, airfoil_polar_files= airfoil.polar_files) 
    else:
        print('\nDefaulting to scaled DAE51') 
        
    while diff>tol:      
        # assign chord distribution
        prop.chord_distribution = c  
                         
        #Things that need a loop
        Tcnew   = Tc     
        tanphit = lamda*(1.+zeta/2.)   # Tangent of the flow angle at the tip
        phit    = np.arctan(tanphit)   # Flow angle at the tip
        tanphi  = tanphit/chi          # Flow angle at every station
        f       = (B/2.)*(1.-chi)/np.sin(phit) 
        F       = (2./np.pi)*np.arccos(np.exp(-f)) #Prandtl momentum loss factor
        phi     = np.arctan(tanphi)    # Flow angle at every station
        
        #Step 3, determine the product Wc, and RE
        G       = F*x*np.cos(phi)*np.sin(phi) #Circulation function
        Wc      = 4.*np.pi*lamda*G*V*R*zeta/(Cl*B)
        Ma      = Wc/speed_of_sound
        RE      = Wc/nu

        if num_airfoils>0:
            # assign initial values 
            alpha0   = np.ones(N)*0.05
            
            # solve for optimal alpha to meet design Cl target
            sol      = root(objective, x0 = alpha0 , args=(airfoils,a_loc,RE,Cl,N))
            alpha    = sol.x
            
            # query surrogate for sectional Cls at stations 
            Cdval    = np.zeros_like(RE) 
            for j,airfoil in enumerate(airfoils):                   
                pd          = airfoil.polars
                Cdval_af    = interp2d(RE,alpha,pd.reynolds_numbers, pd.angle_of_attacks, pd.drag_coefficients)
                locs        = np.where(np.array(a_loc) == j )
                Cdval[locs] = Cdval_af[locs]    
                
        else:    
            Cdval   = (0.108*(Cl**4)-0.2612*(Cl**3)+0.181*(Cl**2)-0.0139*Cl+0.0278)*((50000./RE)**0.2)
            alpha   = Cl/(2.*np.pi)
            
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
    if num_airfoils>0:
        for j,airfoil in enumerate(airfoils): 
            a_geo         = airfoil.geometry
            locs          = np.where(np.array(a_loc) == j )
            t_max[locs]   = a_geo.max_thickness*c[locs] 
            t_c[locs]     = a_geo.thickness_to_chord 
    else:     
        c_blade = np.repeat(np.atleast_2d(np.linspace(0,1,N)),N, axis = 0)* np.repeat(np.atleast_2d(c).T,N, axis = 1)
        t       = (5*c_blade)*(0.2969*np.sqrt(c_blade) - 0.1260*c_blade - 0.3516*(c_blade**2) + 0.2843*(c_blade**3) - 0.1015*(c_blade**4)) # local thickness distribution
        t_max   = np.max(t,axis = 1) 
        t_c     = np.max(t,axis = 1) /c  
            
    # Nondimensional thrust
    if prop.cruise.design_power == None: 
        prop.cruise.design_power = Power[0]        
    elif prop.cruise.design_thrust == None: 
        prop.cruise.design_thrust = Thrust[0]       
    
    # blade solidity
    r          = chi*R                    # Radial coordinate   
    blade_area = sp.integrate.cumtrapz(B*c, r-r[0])
    sigma      = blade_area[-1]/(np.pi*R**2)   
    
    prop.cruise.design_torque                   = Power[0]/omega
    prop.max_thickness_distribution             = t_max
    prop.twist_distribution                     = beta
    prop.chord_distribution                     = c
    prop.radius_distribution                    = r
    prop.number_of_blades                       = int(B)
    prop.cruise.design_power_coefficient        = Cp
    prop.cruise.design_thrust_coefficient       = Ct
    prop.mid_chord_alignment                    = MCA
    prop.thickness_to_chord                     = t_c
    prop.blade_solidity                         = sigma

    return prop

    
def objective(x,airfoils,a_loc,RE,Cl,N):
    # query surrogate for sectional Cls at stations 
    Cl_vals          = np.zeros(N)       
    for j,airfoil in enumerate(airfoils): 
        pd            = airfoil.polars
        Cl_af         = interp2d(RE,x,pd.reynolds_numbers, pd.angle_of_attacks, pd.lift_coefficients)
        locs          = np.where(np.array(a_loc) == j )
        Cl_vals[locs] = Cl_af[locs] 
        
    # compute Cl residual    
    Cl_residuals = Cl_vals - Cl 
    return  Cl_residuals 
