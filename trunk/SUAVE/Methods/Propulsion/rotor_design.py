## @ingroup Methods-Propulsion
# rotor_design.py
# 
# Created: May 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
import numpy as np
import scipy as sp
from SUAVE.Core import Units , Data
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry \
     import import_airfoil_geometry
from scipy.optimize import minimize 
from scipy.optimize import NonlinearConstraint
from scipy import interpolate
import scipy as sp
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_polars \
     import compute_airfoil_polars
# ----------------------------------------------------------------------
#  Propeller Design
# ----------------------------------------------------------------------

def rotor_design(prop, number_of_stations = 20,
                 design_taper             = 0.3,
                 pivot_points             = [0.25,0.5,0.75],
                 linear_interp_flag       = True,
                 #solver                  = 'SLSQP',
                 solver                   = 'differential_evolution',
                 CL_max                   = 1.1,
                 design_FM                = 0.75,
                 print_iter               = True):
    
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
    N             = number_of_stations       # this number determines the discretization of the propeller into stations 
    B             = prop.number_of_blades
    R             = prop.tip_radius
    Rh            = prop.hub_radius
    omega         = prop.angular_velocity    # Rotation Rate in rad/s
    Va            = prop.induced_hover_velocity
    Vinf          = prop.freestream_velocity # Freestream Velocity 
    alt           = prop.design_altitude
    design_thrust = prop.design_thrust
    design_power  = prop.design_power
    a_geo         = prop.airfoil_geometry
    a_pol         = prop.airfoil_polars        
    a_loc         = prop.airfoil_polar_stations    
    
    if (design_thrust == None) and (design_power== None):
        raise AssertionError('Specify either design thrust or design power!')
    
    elif (design_thrust!= None) and (design_power!= None):
        raise AssertionError('Specify either design thrust or design power!')
    
    if prop.rotation == None:
        prop.rotation = list(np.ones(int(B))) 
        
    if  a_pol != None and a_loc != None:
        if len(a_loc) != N:
            raise AssertionError('\nDimension of airfoil sections must be equal to number of stations on rotor')
        # compute airfoil polars for airfoils 
        airfoil_polars  = compute_airfoil_polars(a_geo, a_pol)  
        cl_sur = airfoil_polars.lift_coefficient_surrogates 
        cd_sur = airfoil_polars.drag_coefficient_surrogates    
        
    else:
        raise AssertionError('\nDefine rotor airfoil') 
        
        
    # Calculated total velocity 
    V       = Vinf + Va
    chi0    = Rh/R # Where the rotor blade actually starts
    chi     = np.linspace(chi0,1,N+1) # Vector of nondimensional radii
    chi     = chi[0:N]
         
    # Calculate atmospheric properties
    atmosphere     = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data      = atmosphere.compute_values(alt) 
    p              = atmo_data.pressure[0]
    T              = atmo_data.temperature[0]
    rho            = atmo_data.density[0]
    a              = atmo_data.speed_of_sound[0]
    mu             = atmo_data.dynamic_viscosity[0]
    nu             = mu/rho  
    V              = np.array(V,ndmin=2)
    omega          = np.array(omega,ndmin=2)
    
    # Define initial conditions 
    total_pivots = len(pivot_points) + 2
    
    # Define bounds  
    c_lb    = 0.05*R    # small bound is 40 % of radius 
    c_ub    = 0.4*R     # upper bound is 40 % of radius 
    beta_lb = 0         #  
    beta_ub = np.pi/4   # upper bound approx. 60 degrees  
    c_bnds    = []
    beta_bnds = []
    for i in range(total_pivots):
        c_bnds.append((c_lb,c_ub)) 
        beta_bnds.append((beta_lb,beta_ub))  
    de_bnds = c_bnds + beta_bnds
    bnds    = tuple(de_bnds)

    # Define initial conditions 
    total_pivots = len(pivot_points) + 2   
    c_0          = np.linspace(0.2,0.4,4)*R # 0.2
    beta_0       = np.linspace(0.2,np.pi/4,4)   # 0.3     
    
    # Define static arguments 
    args = (B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,total_pivots,pivot_points,
            linear_interp_flag,design_thrust,design_power,design_taper,CL_max,design_FM)  
    
    init_scaling = np.linspace(0.2,1,3) 
    success_flag = False 
    for i_ in range(len(init_scaling)): 
        #initials      = list(np.concatenate([np.ones(total_pivots)*c_0[i_],np.ones(total_pivots)*beta_0[i_]]))   
        initials      = np.concatenate([np.linspace(0.4,0.1,total_pivots),np.linspace(0.8,0.4,total_pivots),])*init_scaling[i_] 
        if solver == 'SLSQP': 
            # Define constaints 
            cons = [{'type':'ineq', 'fun': constraint_thrust_power   ,'args': args},
                    {'type':'ineq', 'fun': constraint_blade_taper    ,'args': args},
                    {'type':'ineq', 'fun': constraint_monotonic_chord,'args': args},
                    {'type':'ineq', 'fun': constraint_monotonic_twist,'args': args},
                    #{'type':'ineq', 'fun': constraint_blade_solidity, 'args': args},
                    {'type':'ineq', 'fun': constraint_max_cl         ,'args': args}, 
                    ] 
            
            opts= {'eps':1e-2,'maxiter': 500, 'disp': print_iter, 'ftol': 1e-4} 
            try:  
                sol = minimize(minimize_objective,initials, args=args,method=solver, bounds=bnds , constraints= cons, options = opts)
                success_flag = sol.success
            except: 
                success_flag = False   
      
        elif solver == 'differential_evolution': 
            opts  ={'eps':1e-2, 'disp':print_iter , 'ftol': 1e-4}
            diff_evo_cons = create_diff_evo_cons(B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                                            total_pivots,pivot_points,linear_interp_flag,design_thrust,
                                            design_power,design_taper,CL_max,design_FM)  
            
            sol = sp.optimize.differential_evolution(minimize_objective, bounds= de_bnds,args= args, strategy='best1bin', maxiter=2000, \
                                                         tol=0.01, mutation=(0.5, 1), recombination=0.7, callback=None,seed = 2,\
                                                         disp=print_iter, polish=True, init='latinhypercube', atol=0, updating='immediate',\
                                                         workers=1,constraints=diff_evo_cons) 
        
            if sol.success == True:
                success_flag = sol.success
            else:
                success_flag = False  
            
        if success_flag:
            break 
            
    print(sol)
    
    mcc  = constraint_monotonic_chord(sol.x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                               total_pivots,pivot_points,linear_interp_flag,design_thrust,
                               design_power,design_taper,CL_max,design_FM)
    print(mcc)
    mtc = constraint_monotonic_twist(sol.x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
               total_pivots,pivot_points,linear_interp_flag,design_thrust,
               design_power,design_taper,CL_max,design_FM)
    
    print(mtc)
    
    
    # Discretized propeller into stations using linear interpolation
    if linear_interp_flag:
        c    = linear_discretize(sol.x[:total_pivots],chi,pivot_points)     
        beta = linear_discretize(sol.x[total_pivots:],chi,pivot_points)  
    else: 
        c    = spline_discretize(sol.x[:total_pivots],chi,pivot_points)     
        beta = spline_discretize(sol.x[total_pivots:],chi,pivot_points) 
    
    # Things that don't change with iteration
    Nr       = len(c) # Number of stations radially    
    ctrl_pts = 1
    BB       = B*B    
    BBB      = BB*B   
    omega    = np.abs(omega)        
    r        = chi*R            # Radial coordinate 
    omegar   = np.outer(omega,r)
    pi       = np.pi            
    pi2      = pi*pi        
    n        = omega/(2.*pi)    # Cycles per second  
    deltar   = (r[1]-r[0])         
    rho_0    = rho

    # Setup a Newton iteration
    diff   = 1. 
    ii     = 0
    tol    = 1e-6  # Convergence tolerance
    
    # uniform freestream
    ua       = np.zeros_like(V)              
    ut       = np.zeros_like(V)             
    ur       = np.zeros_like(V)

    # total velocities
    Ua     = np.outer((V + ua),np.ones_like(r)) 

    # Things that will change with iteration
    size   = (ctrl_pts,Nr)
    PSI    = np.ones(size)
    PSIold = np.zeros(size)  

    # total velocities
    Ut   = omegar - ut
    U    = np.sqrt(Ua*Ua + Ut*Ut + ur*ur)

    # Drela's Theory
    while (diff>tol):
        sin_psi      = np.sin(PSI)
        cos_psi      = np.cos(PSI)
        Wa           = 0.5*Ua + 0.5*U*sin_psi
        Wt           = 0.5*Ut + 0.5*U*cos_psi   
        va           = Wa - Ua
        vt           = Ut - Wt
        alpha        = beta - np.arctan2(Wa,Wt)
        W            = (Wa*Wa + Wt*Wt)**0.5
        Ma           = (W)/a        # a is the speed of sound  
        lamdaw       = r*Wa/(R*Wt) 

        # Limiter to keep from Nan-ing
        lamdaw[lamdaw<0.] = 0. 
        f            = (B/2.)*(1.-r/R)/lamdaw
        f[f<0.]      = 0.
        piece        = np.exp(-f)
        arccos_piece = np.arccos(piece)
        F            = 2.*arccos_piece/pi # Prandtl's tip factor
        Gamma        = vt*(4.*pi*r/B)*F*(1.+(4.*lamdaw*R/(pi*B*r))*(4.*lamdaw*R/(pi*B*r)))**0.5 
        Re           = (W*c)/nu  

        # Compute aerodynamic forces based on specified input airfoil or using a surrogate
        Cl, Cdval = compute_aerodynamic_forces(a_loc, a_geo, cl_sur, cd_sur, ctrl_pts, Nr, Re, Ma, alpha)

        Rsquiggly   = Gamma - 0.5*W*c*Cl

        # An analytical derivative for dR_dpsi, this is derived by taking a derivative of the above equations
        # This was solved symbolically in Matlab and exported        
        f_wt_2      = 4*Wt*Wt
        f_wa_2      = 4*Wa*Wa
        Ucospsi     = U*cos_psi
        Usinpsi     = U*sin_psi
        Utcospsi    = Ut*cos_psi
        Uasinpsi    = Ua*sin_psi 
        UapUsinpsi  = (Ua + Usinpsi)
        utpUcospsi  = (Ut + Ucospsi) 
        utpUcospsi2 = utpUcospsi*utpUcospsi
        UapUsinpsi2 = UapUsinpsi*UapUsinpsi 
        dR_dpsi     = ((4.*U*r*arccos_piece*sin_psi*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5))/B - 
                           (pi*U*(Ua*cos_psi - Ut*sin_psi)*(beta - np.arctan((Wa+Wa)/(Wt+Wt))))/(2.*(f_wt_2 + f_wa_2)**(0.5))
                           + (pi*U*(f_wt_2 +f_wa_2)**(0.5)*(U + Utcospsi  +  Uasinpsi))/(2.*(f_wa_2/(f_wt_2) + 1.)*utpUcospsi2)
                           - (4.*U*piece*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5)*(R - r)*(Ut/2. - 
                            (Ucospsi)/2.)*(U + Utcospsi + Uasinpsi ))/(f_wa_2*(1. - np.exp(-(B*(Wt+Wt)*(R - 
                            r))/(r*(Wa+Wa))))**(0.5)) + (128.*U*r*arccos_piece*(Wa+Wa)*(Ut/2. - (Ucospsi)/2.)*(U + 
                            Utcospsi  + Uasinpsi ))/(BBB*pi2*utpUcospsi*utpUcospsi2*((16.*f_wa_2)/(BB*pi2*f_wt_2) + 1.)**(0.5))) 

        dR_dpsi[np.isnan(dR_dpsi)] = 0.1

        dpsi        = -Rsquiggly/dR_dpsi
        PSI         = PSI + dpsi
        diff        = np.max(abs(PSIold-PSI))
        PSIold      = PSI

        # omega = 0, do not run BEMT convergence loop 
        if all(omega[:,0]) == 0. :
            break

        # If its really not going to converge
        if np.any(PSI>pi/2) and np.any(dpsi>0.0):
            break

        ii+=1 
        if ii>10000:
            break

    # More Cd scaling from Mach from AA241ab notes for turbulent skin friction
    Tw_Tinf     = 1. + 1.78*(Ma*Ma)
    Tp_Tinf     = 1. + 0.035*(Ma*Ma) + 0.45*(Tw_Tinf-1.)
    Tp          = (Tp_Tinf)*T
    Rp_Rinf     = (Tp_Tinf**2.5)*(Tp+110.4)/(T+110.4) 
    Cd          = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval  

    epsilon                  = Cd/Cl
    epsilon[epsilon==np.inf] = 10. 

    blade_T_distribution    = rho*(Gamma*(Wt-epsilon*Wa))*deltar 
    blade_Q_distribution    = rho*(Gamma*(Wa+epsilon*Wt)*r)*deltar 
    thrust                  = np.atleast_2d((B * np.sum(blade_T_distribution, axis = 1))).T 
    torque                  = np.atleast_2d((B * np.sum(blade_Q_distribution, axis = 1))).T         
    power                   = omega*torque   

    # calculate coefficients 
    D        = 2*R 
    Cq       = torque/(rho_0*(n*n)*(D*D*D*D*D)) 
    Ct       = thrust/(rho_0*(n*n)*(D*D*D*D))
    Cp       = power/(rho_0*(n*n*n)*(D*D*D*D*D))
    etap     = V*thrust/power  
     
    if prop.design_power == None: 
        prop.design_power = power[0][0]
        
    design_torque = power[0][0]/omega[0][0]
    
    # blade solidity
    r          = chi*R                    # Radial coordinate   
    blade_area = sp.integrate.cumtrapz(B*c, r-r[0])
    sigma      = blade_area[-1]/(np.pi*R**2)   
    
    MCA    = c/4. - c[0]/4.
    airfoil_geometry_data = import_airfoil_geometry(a_geo) 
    t_max = np.take(airfoil_geometry_data.max_thickness,a_loc,axis=0)*c 
    t_c   =  np.take(airfoil_geometry_data.thickness_to_chord,a_loc,axis=0)  
    
    prop.design_torque              = design_torque
    prop.max_thickness_distribution = t_max
    prop.twist_distribution         = beta
    prop.chord_distribution         = c
    prop.radius_distribution        = r 
    prop.number_of_blades           = int(B) 
    prop.design_power_coefficient   = Cp[0][0] 
    prop.design_thrust_coefficient  = Ct[0][0] 
    prop.mid_chord_alignment        = MCA
    prop.thickness_to_chord         = t_c 
    prop.blade_solidity             = sigma  
    prop.airfoil_cl_surrogates      = cl_sur
    prop.airfoil_cd_surrogates      = cd_sur 
    prop.airfoil_flag               = True 

    return prop
  

def create_diff_evo_cons(B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                       total_pivots,pivot_points,linear_interp_flag,design_thrust,
                       design_power,design_taper,CL_max,design_FM):
    constraints = []
    def fun1(x): 
        con1 = constraint_thrust_power(x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                                               total_pivots,pivot_points,linear_interp_flag,design_thrust,
                                            design_power,design_taper,CL_max,design_FM)

        return np.atleast_1d(con1)

    def fun2(x): 
        con2 = constraint_blade_taper(x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                                              total_pivots,pivot_points,linear_interp_flag,design_thrust,
                                            design_power,design_taper,CL_max,design_FM)

        return np.atleast_1d(con2)


    def fun3(x): 
        con3 = constraint_monotonic_chord(x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                                                  total_pivots,pivot_points,linear_interp_flag,design_thrust,
                                            design_power,design_taper,CL_max,design_FM)

        return np.atleast_1d(con3)


    def fun4(x): 
        con4 = constraint_monotonic_twist(x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                                                  total_pivots,pivot_points,linear_interp_flag,design_thrust,
                                            design_power,design_taper,CL_max,design_FM)

        return np.atleast_1d(con4)

    def fun5(x): 
        con5 = constraint_blade_solidity(x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                                                 total_pivots,pivot_points,linear_interp_flag,design_thrust,
                                            design_power,design_taper,CL_max,design_FM)
        return np.atleast_1d(con5)

    def fun6(x): 
        con6 = constraint_figure_of_merit(x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                                                 total_pivots,pivot_points,linear_interp_flag,design_thrust,
                                            design_power,design_taper,CL_max,design_FM)
        return np.atleast_1d(con6)

    def fun7(x): 
        con7 = constraint_max_cl(x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                                                 total_pivots,pivot_points,linear_interp_flag,design_thrust,
                                            design_power,design_taper,CL_max,design_FM)


        return np.atleast_1d(con7)            

    constraints.append(NonlinearConstraint(fun1, 0 ,np.inf)) 
    constraints.append(NonlinearConstraint(fun2, 0 ,np.inf))  
    constraints.append(NonlinearConstraint(fun3, 0 ,np.inf))  
    constraints.append(NonlinearConstraint(fun4, 0 ,np.inf))
    #constraints.append(NonlinearConstraint(fun5, 0 ,np.inf)) 
    #constraints.append(NonlinearConstraint(fun6, 0 ,np.inf)) 
    constraints.append(NonlinearConstraint(fun7, 0 ,np.inf)) 
    
    return tuple(constraints)     

# objective function
def minimize_objective(x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                       total_pivots,pivot_points,linear_interp_flag,design_thrust,
                       design_power,design_taper,CL_max,design_FM): 

    # Discretized propeller into stations using linear interpolation
    if linear_interp_flag:
        c    = linear_discretize(x[:total_pivots],chi,pivot_points)     
        beta = linear_discretize(x[total_pivots:],chi,pivot_points)  
    else: 
        c    = spline_discretize(x[:total_pivots],chi,pivot_points)     
        beta = spline_discretize(x[total_pivots:],chi,pivot_points) 
    
    # Things that don't change with iteration
    Nr       = len(c) # Number of stations radially    
    ctrl_pts = 1
    BB       = B*B    
    BBB      = BB*B   
    omega    = np.abs(omega)        
    r        = chi*R            # Radial coordinate 
    omegar   = np.outer(omega,r)
    pi       = np.pi            
    pi2      = pi*pi        
    n        = omega/(2.*pi)    # Cycles per second  
    deltar   = (r[1]-r[0])         
    rho_0    = rho

    # Setup a Newton iteration
    diff   = 1. 
    ii     = 0
    tol    = 1e-6  # Convergence tolerance
    
    # uniform freestream
    ua       = np.zeros_like(V)              
    ut       = np.zeros_like(V)             
    ur       = np.zeros_like(V)

    # total velocities
    Ua     = np.outer((V + ua),np.ones_like(r)) 

    # Things that will change with iteration
    size   = (ctrl_pts,Nr)
    PSI    = np.ones(size)
    PSIold = np.zeros(size)  

    # total velocities
    Ut   = omegar - ut
    U    = np.sqrt(Ua*Ua + Ut*Ut + ur*ur)

    # Drela's Theory
    while (diff>tol):
        sin_psi      = np.sin(PSI)
        cos_psi      = np.cos(PSI)
        Wa           = 0.5*Ua + 0.5*U*sin_psi
        Wt           = 0.5*Ut + 0.5*U*cos_psi   
        va           = Wa - Ua
        vt           = Ut - Wt
        alpha        = beta - np.arctan2(Wa,Wt)
        W            = (Wa*Wa + Wt*Wt)**0.5
        Ma           = (W)/a        # a is the speed of sound  
        lamdaw       = r*Wa/(R*Wt) 

        # Limiter to keep from Nan-ing
        lamdaw[lamdaw<0.] = 0. 
        f            = (B/2.)*(1.-r/R)/lamdaw
        f[f<0.]      = 0.
        piece        = np.exp(-f)
        arccos_piece = np.arccos(piece)
        F            = 2.*arccos_piece/pi # Prandtl's tip factor
        Gamma        = vt*(4.*pi*r/B)*F*(1.+(4.*lamdaw*R/(pi*B*r))*(4.*lamdaw*R/(pi*B*r)))**0.5 
        Re           = (W*c)/nu  

        # Compute aerodynamic forces based on specified input airfoil or using a surrogate
        Cl, Cdval = compute_aerodynamic_forces(a_loc, a_geo, cl_sur, cd_sur, ctrl_pts, Nr, Re, Ma, alpha)

        Rsquiggly   = Gamma - 0.5*W*c*Cl

        # An analytical derivative for dR_dpsi, this is derived by taking a derivative of the above equations
        # This was solved symbolically in Matlab and exported        
        f_wt_2      = 4*Wt*Wt
        f_wa_2      = 4*Wa*Wa
        Ucospsi     = U*cos_psi
        Usinpsi     = U*sin_psi
        Utcospsi    = Ut*cos_psi
        Uasinpsi    = Ua*sin_psi 
        UapUsinpsi  = (Ua + Usinpsi)
        utpUcospsi  = (Ut + Ucospsi) 
        utpUcospsi2 = utpUcospsi*utpUcospsi
        UapUsinpsi2 = UapUsinpsi*UapUsinpsi 
        dR_dpsi     = ((4.*U*r*arccos_piece*sin_psi*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5))/B - 
                           (pi*U*(Ua*cos_psi - Ut*sin_psi)*(beta - np.arctan((Wa+Wa)/(Wt+Wt))))/(2.*(f_wt_2 + f_wa_2)**(0.5))
                           + (pi*U*(f_wt_2 +f_wa_2)**(0.5)*(U + Utcospsi  +  Uasinpsi))/(2.*(f_wa_2/(f_wt_2) + 1.)*utpUcospsi2)
                           - (4.*U*piece*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5)*(R - r)*(Ut/2. - 
                            (Ucospsi)/2.)*(U + Utcospsi + Uasinpsi ))/(f_wa_2*(1. - np.exp(-(B*(Wt+Wt)*(R - 
                            r))/(r*(Wa+Wa))))**(0.5)) + (128.*U*r*arccos_piece*(Wa+Wa)*(Ut/2. - (Ucospsi)/2.)*(U + 
                            Utcospsi  + Uasinpsi ))/(BBB*pi2*utpUcospsi*utpUcospsi2*((16.*f_wa_2)/(BB*pi2*f_wt_2) + 1.)**(0.5))) 

        dR_dpsi[np.isnan(dR_dpsi)] = 0.1

        dpsi        = -Rsquiggly/dR_dpsi
        PSI         = PSI + dpsi
        diff        = np.max(abs(PSIold-PSI))
        PSIold      = PSI

        # omega = 0, do not run BEMT convergence loop 
        if all(omega[:,0]) == 0. :
            break

        # If its really not going to converge
        if np.any(PSI>pi/2) and np.any(dpsi>0.0):
            break

        ii+=1 
        if ii>10000:
            break

    # More Cd scaling from Mach from AA241ab notes for turbulent skin friction
    Tw_Tinf     = 1. + 1.78*(Ma*Ma)
    Tp_Tinf     = 1. + 0.035*(Ma*Ma) + 0.45*(Tw_Tinf-1.)
    Tp          = (Tp_Tinf)*T
    Rp_Rinf     = (Tp_Tinf**2.5)*(Tp+110.4)/(T+110.4) 
    Cd          = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval  

    epsilon                  = Cd/Cl
    epsilon[epsilon==np.inf] = 10. 

    blade_T_distribution    = rho*(Gamma*(Wt-epsilon*Wa))*deltar 
    blade_Q_distribution    = rho*(Gamma*(Wa+epsilon*Wt)*r)*deltar 
    thrust                  = np.atleast_2d((B * np.sum(blade_T_distribution, axis = 1))).T 
    torque                  = np.atleast_2d((B * np.sum(blade_Q_distribution, axis = 1))).T         
    power                   = omega*torque   

    # calculate coefficients 
    D        = 2*R 
    Cq       = torque/(rho_0*(n*n)*(D*D*D*D*D)) 
    Ct       = thrust/(rho_0*(n*n)*(D*D*D*D))
    Cp       = power/(rho_0*(n*n*n)*(D*D*D*D*D))
    etap     = V*thrust/power  
    
    if design_thrust == None:
        return power[0][0]
    
    if design_power == None:
        return thrust[0][0]
    
    return 

def compute_aerodynamic_forces(a_loc, a_geo, cl_sur, cd_sur, ctrl_pts, Nr, Re, Ma, alpha):
    """
    Cl, Cdval = compute_aerodynamic_forces(  a_loc, 
                                             a_geo, 
                                             cl_sur, 
                                             cd_sur, 
                                             ctrl_pts, 
                                             Nr, 
                                             Na, 
                                             Re, 
                                             Ma, 
                                             alpha, 
                                             tc, 
                                             nonuniform_freestream )
                                             
    Computes the aerodynamic forces at sectional blade locations. If airfoil 
    geometry and locations are specified, the forces are computed using the 
    airfoil polar lift and drag surrogates, accounting for the local Reynolds 
    number and local angle of attack. 
    
    If the airfoils are not specified, an approximation is used.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    a_loc                      Locations of specified airfoils                 [-]
    a_geo                      Geometry of specified airfoil                   [-]
    cl_sur                     Lift Coefficient Surrogates                     [-]
    cd_sur                     Drag Coefficient Surrogates                     [-]
    ctrl_pts                   Number of control points                        [-]
    Nr                         Number of radial blade sections                 [-]
    Re                         Local Reynolds numbers                          [-]
    Ma                         Local Mach number                               [-]
    alpha                      Local angles of attack                          [radians]
   
                                                     
                                                     
    Outputs:                                          
    Cl                       Lift Coefficients                         [-]                               
    Cdval                    Drag Coefficients  (before scaling)       [-]
    """         
    # Compute blade Cl and Cd distribution from the airfoil data  
    dim_sur = len(cl_sur)    
    # return the 1D Cl and CDval of shape (ctrl_pts, Nr)
    Cl      = np.zeros((ctrl_pts,Nr))              
    Cdval   = np.zeros((ctrl_pts,Nr))  
    
    for jj in range(dim_sur):                 
        Cl_af         = cl_sur[a_geo[jj]](Re,alpha,grid=False)  
        Cdval_af      = cd_sur[a_geo[jj]](Re,alpha,grid=False)  
        locs          = np.where(np.array(a_loc) == jj )
        Cl[:,locs]    = Cl_af[:,locs]
        Cdval[:,locs] = Cdval_af[:,locs]             
    
    return Cl, Cdval

 
def constraint_thrust_power(x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                            total_pivots,pivot_points,linear_interp_flag,design_thrust,
                            design_power,design_taper,CL_max,design_FM): 
    
    # Discretized propeller into stations using linear interpolation 
    if linear_interp_flag:
        c    = linear_discretize(x[:total_pivots],chi,pivot_points)     
        beta = linear_discretize(x[total_pivots:],chi,pivot_points)  
    else: 
        c    = spline_discretize(x[:total_pivots],chi,pivot_points)     
        beta = spline_discretize(x[total_pivots:],chi,pivot_points) 
    
    # Things that don't change with iteration
    Nr       = len(c) # Number of stations radially    
    ctrl_pts = 1
    BB       = B*B    
    BBB      = BB*B   
    omega    = np.abs(omega)        
    r        = chi*R            # Radial coordinate 
    omegar   = np.outer(omega,r)
    pi       = np.pi            
    pi2      = pi*pi        
    n        = omega/(2.*pi)    # Cycles per second  
    deltar   = (r[1]-r[0])         
    rho_0    = rho

    # Setup a Newton iteration
    diff   = 1. 
    ii     = 0
    tol    = 1e-6  # Convergence tolerance
    
    # uniform freestream
    ua       = np.zeros_like(V)              
    ut       = np.zeros_like(V)             
    ur       = np.zeros_like(V)

    # total velocities
    Ua     = np.outer((V + ua),np.ones_like(r)) 

    # Things that will change with iteration
    size   = (ctrl_pts,Nr)
    PSI    = np.ones(size)
    PSIold = np.zeros(size)  

    # total velocities
    Ut   = omegar - ut
    U    = np.sqrt(Ua*Ua + Ut*Ut + ur*ur)

    # Drela's Theory
    while (diff>tol):
        sin_psi      = np.sin(PSI)
        cos_psi      = np.cos(PSI)
        Wa           = 0.5*Ua + 0.5*U*sin_psi
        Wt           = 0.5*Ut + 0.5*U*cos_psi   
        va           = Wa - Ua
        vt           = Ut - Wt
        alpha        = beta - np.arctan2(Wa,Wt)
        W            = (Wa*Wa + Wt*Wt)**0.5
        Ma           = (W)/a        # a is the speed of sound  
        lamdaw       = r*Wa/(R*Wt) 

        # Limiter to keep from Nan-ing
        lamdaw[lamdaw<0.] = 0. 
        f            = (B/2.)*(1.-r/R)/lamdaw
        f[f<0.]      = 0.
        piece        = np.exp(-f)
        arccos_piece = np.arccos(piece)
        F            = 2.*arccos_piece/pi # Prandtl's tip factor
        Gamma        = vt*(4.*pi*r/B)*F*(1.+(4.*lamdaw*R/(pi*B*r))*(4.*lamdaw*R/(pi*B*r)))**0.5 
        Re           = (W*c)/nu  

        # Compute aerodynamic forces based on specified input airfoil or using a surrogate
        Cl, Cdval = compute_aerodynamic_forces(a_loc, a_geo, cl_sur, cd_sur, ctrl_pts, Nr, Re, Ma, alpha)

        Rsquiggly   = Gamma - 0.5*W*c*Cl

        # An analytical derivative for dR_dpsi, this is derived by taking a derivative of the above equations
        # This was solved symbolically in Matlab and exported        
        f_wt_2      = 4*Wt*Wt
        f_wa_2      = 4*Wa*Wa
        Ucospsi     = U*cos_psi
        Usinpsi     = U*sin_psi
        Utcospsi    = Ut*cos_psi
        Uasinpsi    = Ua*sin_psi 
        UapUsinpsi  = (Ua + Usinpsi)
        utpUcospsi  = (Ut + Ucospsi) 
        utpUcospsi2 = utpUcospsi*utpUcospsi
        UapUsinpsi2 = UapUsinpsi*UapUsinpsi 
        dR_dpsi     = ((4.*U*r*arccos_piece*sin_psi*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5))/B - 
                           (pi*U*(Ua*cos_psi - Ut*sin_psi)*(beta - np.arctan((Wa+Wa)/(Wt+Wt))))/(2.*(f_wt_2 + f_wa_2)**(0.5))
                           + (pi*U*(f_wt_2 +f_wa_2)**(0.5)*(U + Utcospsi  +  Uasinpsi))/(2.*(f_wa_2/(f_wt_2) + 1.)*utpUcospsi2)
                           - (4.*U*piece*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5)*(R - r)*(Ut/2. - 
                            (Ucospsi)/2.)*(U + Utcospsi + Uasinpsi ))/(f_wa_2*(1. - np.exp(-(B*(Wt+Wt)*(R - 
                            r))/(r*(Wa+Wa))))**(0.5)) + (128.*U*r*arccos_piece*(Wa+Wa)*(Ut/2. - (Ucospsi)/2.)*(U + 
                            Utcospsi  + Uasinpsi ))/(BBB*pi2*utpUcospsi*utpUcospsi2*((16.*f_wa_2)/(BB*pi2*f_wt_2) + 1.)**(0.5))) 

        dR_dpsi[np.isnan(dR_dpsi)] = 0.1

        dpsi        = -Rsquiggly/dR_dpsi
        PSI         = PSI + dpsi
        diff        = np.max(abs(PSIold-PSI))
        PSIold      = PSI

        # omega = 0, do not run BEMT convergence loop 
        if all(omega[:,0]) == 0. :
            break

        # If its really not going to converge
        if np.any(PSI>pi/2) and np.any(dpsi>0.0):
            break

        ii+=1 
        if ii>10000:
            break

    # More Cd scaling from Mach from AA241ab notes for turbulent skin friction
    Tw_Tinf     = 1. + 1.78*(Ma*Ma)
    Tp_Tinf     = 1. + 0.035*(Ma*Ma) + 0.45*(Tw_Tinf-1.)
    Tp          = (Tp_Tinf)*T
    Rp_Rinf     = (Tp_Tinf**2.5)*(Tp+110.4)/(T+110.4) 
    Cd          = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval  

    epsilon                  = Cd/Cl
    epsilon[epsilon==np.inf] = 10. 

    blade_T_distribution    = rho*(Gamma*(Wt-epsilon*Wa))*deltar 
    blade_Q_distribution    = rho*(Gamma*(Wa+epsilon*Wt)*r)*deltar 
    thrust                  = np.atleast_2d((B * np.sum(blade_T_distribution, axis = 1))).T 
    torque                  = np.atleast_2d((B * np.sum(blade_Q_distribution, axis = 1))).T         
    power                   = omega*torque   

    # calculate coefficients 
    D        = 2*R 
    Cq       = torque/(rho_0*(n*n)*(D*D*D*D*D)) 
    Ct       = thrust/(rho_0*(n*n)*(D*D*D*D))
    Cp       = power/(rho_0*(n*n*n)*(D*D*D*D*D))
    etap     = V*thrust/power  
    
    if design_thrust == None:
        constraint = 0.005*design_thrust - abs(power[0][0] - design_power)    # error bound = 0.5 % 
    
    if design_power == None:
        constraint = 0.005*design_thrust - abs(thrust[0][0] - design_thrust)   
        
    return  constraint


def constraint_max_cl(x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                            total_pivots,pivot_points,linear_interp_flag,design_thrust,
                            design_power,design_taper,CL_max,design_FM): 
    
    # Discretized propeller into stations using linear interpolation 
    if linear_interp_flag:
        c    = linear_discretize(x[:total_pivots],chi,pivot_points)     
        beta = linear_discretize(x[total_pivots:],chi,pivot_points)  
    else: 
        c    = spline_discretize(x[:total_pivots],chi,pivot_points)     
        beta = spline_discretize(x[total_pivots:],chi,pivot_points) 
    
    # Things that don't change with iteration
    Nr       = len(c) # Number of stations radially    
    ctrl_pts = 1
    BB       = B*B    
    BBB      = BB*B   
    omega    = np.abs(omega)        
    r        = chi*R            # Radial coordinate 
    omegar   = np.outer(omega,r)
    pi       = np.pi            
    pi2      = pi*pi        
    n        = omega/(2.*pi)    # Cycles per second  
    deltar   = (r[1]-r[0])         
    rho_0    = rho

    # Setup a Newton iteration
    diff   = 1. 
    ii     = 0
    tol    = 1e-6  # Convergence tolerance
    
    # uniform freestream
    ua       = np.zeros_like(V)              
    ut       = np.zeros_like(V)             
    ur       = np.zeros_like(V)

    # total velocities
    Ua     = np.outer((V + ua),np.ones_like(r)) 

    # Things that will change with iteration
    size   = (ctrl_pts,Nr)
    PSI    = np.ones(size)
    PSIold = np.zeros(size)  

    # total velocities
    Ut   = omegar - ut
    U    = np.sqrt(Ua*Ua + Ut*Ut + ur*ur)

    # Drela's Theory
    while (diff>tol):
        sin_psi      = np.sin(PSI)
        cos_psi      = np.cos(PSI)
        Wa           = 0.5*Ua + 0.5*U*sin_psi
        Wt           = 0.5*Ut + 0.5*U*cos_psi   
        va           = Wa - Ua
        vt           = Ut - Wt
        alpha        = beta - np.arctan2(Wa,Wt)
        W            = (Wa*Wa + Wt*Wt)**0.5
        Ma           = (W)/a        # a is the speed of sound  
        lamdaw       = r*Wa/(R*Wt) 

        # Limiter to keep from Nan-ing
        lamdaw[lamdaw<0.] = 0. 
        f            = (B/2.)*(1.-r/R)/lamdaw
        f[f<0.]      = 0.
        piece        = np.exp(-f)
        arccos_piece = np.arccos(piece)
        F            = 2.*arccos_piece/pi # Prandtl's tip factor
        Gamma        = vt*(4.*pi*r/B)*F*(1.+(4.*lamdaw*R/(pi*B*r))*(4.*lamdaw*R/(pi*B*r)))**0.5 
        Re           = (W*c)/nu  

        # Compute aerodynamic forces based on specified input airfoil or using a surrogate
        Cl, Cdval = compute_aerodynamic_forces(a_loc, a_geo, cl_sur, cd_sur, ctrl_pts, Nr, Re, Ma, alpha)

        Rsquiggly   = Gamma - 0.5*W*c*Cl

        # An analytical derivative for dR_dpsi, this is derived by taking a derivative of the above equations
        # This was solved symbolically in Matlab and exported        
        f_wt_2      = 4*Wt*Wt
        f_wa_2      = 4*Wa*Wa
        Ucospsi     = U*cos_psi
        Usinpsi     = U*sin_psi
        Utcospsi    = Ut*cos_psi
        Uasinpsi    = Ua*sin_psi 
        UapUsinpsi  = (Ua + Usinpsi)
        utpUcospsi  = (Ut + Ucospsi) 
        utpUcospsi2 = utpUcospsi*utpUcospsi
        UapUsinpsi2 = UapUsinpsi*UapUsinpsi 
        dR_dpsi     = ((4.*U*r*arccos_piece*sin_psi*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5))/B - 
                           (pi*U*(Ua*cos_psi - Ut*sin_psi)*(beta - np.arctan((Wa+Wa)/(Wt+Wt))))/(2.*(f_wt_2 + f_wa_2)**(0.5))
                           + (pi*U*(f_wt_2 +f_wa_2)**(0.5)*(U + Utcospsi  +  Uasinpsi))/(2.*(f_wa_2/(f_wt_2) + 1.)*utpUcospsi2)
                           - (4.*U*piece*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5)*(R - r)*(Ut/2. - 
                            (Ucospsi)/2.)*(U + Utcospsi + Uasinpsi ))/(f_wa_2*(1. - np.exp(-(B*(Wt+Wt)*(R - 
                            r))/(r*(Wa+Wa))))**(0.5)) + (128.*U*r*arccos_piece*(Wa+Wa)*(Ut/2. - (Ucospsi)/2.)*(U + 
                            Utcospsi  + Uasinpsi ))/(BBB*pi2*utpUcospsi*utpUcospsi2*((16.*f_wa_2)/(BB*pi2*f_wt_2) + 1.)**(0.5))) 

        dR_dpsi[np.isnan(dR_dpsi)] = 0.1

        dpsi        = -Rsquiggly/dR_dpsi
        PSI         = PSI + dpsi
        diff        = np.max(abs(PSIold-PSI))
        PSIold      = PSI

        # omega = 0, do not run BEMT convergence loop 
        if all(omega[:,0]) == 0. :
            break

        # If its really not going to converge
        if np.any(PSI>pi/2) and np.any(dpsi>0.0):
            break

        ii+=1 
        if ii>10000:
            break

    # Cl constraint 
    cl_diff = CL_max - Cl
    CL_constraint = sum(cl_diff[cl_diff<0])*10  
    return CL_constraint


def constraint_blade_taper(x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                           total_pivots,pivot_points,linear_interp_flag,design_thrust,
                           design_power,design_taper,CL_max,design_FM): 
    blade_taper = x[total_pivots-1]/x[0]
    taper_con   = blade_taper - design_taper
    
    return  taper_con


def constraint_monotonic_chord(x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                           total_pivots,pivot_points,linear_interp_flag,design_thrust,
                           design_power,design_taper,CL_max,design_FM):  
  
    violation = 0
    for pi in range(total_pivots-1):
        if (x[pi] - x[pi+1]) < 0:
            violation += (x[pi] - x[pi+1])*10 
            
    return  violation 

def constraint_monotonic_twist(x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                           total_pivots,pivot_points,linear_interp_flag,design_thrust,
                           design_power,design_taper,CL_max,design_FM):  
    violation = 0
    for pi in range(total_pivots-1):
        if ( x[total_pivots+ pi] - x[total_pivots + pi+1]) < 0:
            violation += ( x[total_pivots+ pi] - x[total_pivots + pi+1])*10 
            
    return  violation


def constraint_blade_solidity(x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                              total_pivots,pivot_points,linear_interp_flag,design_thrust,
                              design_power,design_taper,CL_max,design_FM): 
    
    # Discretized propeller into stations using linear interpolation
    c    = linear_discretize(x[:total_pivots],chi,pivot_points)     
    
    # blade solidity
    r          = chi*R                    
    blade_area = sp.integrate.cumtrapz(B*c, r-r[0])
    sigma      = blade_area[-1]/(np.pi*R**2)   
    
    sigma_con  = 0.2 - sigma  # solidity no greater than 20% - typically 5-20% from Ananth
    return sigma_con 


def constraint_figure_of_merit(x,B, R,chi,Rh,a_geo ,a_loc,cl_sur,cd_sur, rho,mu,nu,a,T,V, omega,
                              total_pivots,pivot_points,linear_interp_flag,design_thrust,
                              design_power,design_taper,CL_max,design_FM): 
    
    # Discretized propeller into stations using linear interpolation 
    if linear_interp_flag:
        c    = linear_discretize(x[:total_pivots],chi,pivot_points)     
        beta = linear_discretize(x[total_pivots:],chi,pivot_points)  
    else: 
        c    = spline_discretize(x[:total_pivots],chi,pivot_points)     
        beta = spline_discretize(x[total_pivots:],chi,pivot_points) 
    
    # Things that don't change with iteration
    Nr       = len(c) # Number of stations radially    
    ctrl_pts = 1
    BB       = B*B    
    BBB      = BB*B   
    omega    = np.abs(omega)        
    r        = chi*R            # Radial coordinate 
    omegar   = np.outer(omega,r)
    pi       = np.pi            
    pi2      = pi*pi        
    n        = omega/(2.*pi)    # Cycles per second  
    deltar   = (r[1]-r[0])         
    rho_0    = rho

    # Setup a Newton iteration
    diff   = 1. 
    ii     = 0
    tol    = 1e-6  # Convergence tolerance
    
    # uniform freestream
    ua       = np.zeros_like(V)              
    ut       = np.zeros_like(V)             
    ur       = np.zeros_like(V)

    # total velocities
    Ua     = np.outer((V + ua),np.ones_like(r)) 

    # Things that will change with iteration
    size   = (ctrl_pts,Nr)
    PSI    = np.ones(size)
    PSIold = np.zeros(size)  

    # total velocities
    Ut   = omegar - ut
    U    = np.sqrt(Ua*Ua + Ut*Ut + ur*ur)

    # Drela's Theory
    while (diff>tol):
        sin_psi      = np.sin(PSI)
        cos_psi      = np.cos(PSI)
        Wa           = 0.5*Ua + 0.5*U*sin_psi
        Wt           = 0.5*Ut + 0.5*U*cos_psi   
        va           = Wa - Ua
        vt           = Ut - Wt
        alpha        = beta - np.arctan2(Wa,Wt)
        W            = (Wa*Wa + Wt*Wt)**0.5
        Ma           = (W)/a        # a is the speed of sound  
        lamdaw       = r*Wa/(R*Wt) 

        # Limiter to keep from Nan-ing
        lamdaw[lamdaw<0.] = 0. 
        f            = (B/2.)*(1.-r/R)/lamdaw
        f[f<0.]      = 0.
        piece        = np.exp(-f)
        arccos_piece = np.arccos(piece)
        F            = 2.*arccos_piece/pi # Prandtl's tip factor
        Gamma        = vt*(4.*pi*r/B)*F*(1.+(4.*lamdaw*R/(pi*B*r))*(4.*lamdaw*R/(pi*B*r)))**0.5 
        Re           = (W*c)/nu  

        # Compute aerodynamic forces based on specified input airfoil or using a surrogate
        Cl, Cdval = compute_aerodynamic_forces(a_loc, a_geo, cl_sur, cd_sur, ctrl_pts, Nr, Re, Ma, alpha)

        Rsquiggly   = Gamma - 0.5*W*c*Cl

        # An analytical derivative for dR_dpsi, this is derived by taking a derivative of the above equations
        # This was solved symbolically in Matlab and exported        
        f_wt_2      = 4*Wt*Wt
        f_wa_2      = 4*Wa*Wa
        Ucospsi     = U*cos_psi
        Usinpsi     = U*sin_psi
        Utcospsi    = Ut*cos_psi
        Uasinpsi    = Ua*sin_psi 
        UapUsinpsi  = (Ua + Usinpsi)
        utpUcospsi  = (Ut + Ucospsi) 
        utpUcospsi2 = utpUcospsi*utpUcospsi
        UapUsinpsi2 = UapUsinpsi*UapUsinpsi 
        dR_dpsi     = ((4.*U*r*arccos_piece*sin_psi*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5))/B - 
                           (pi*U*(Ua*cos_psi - Ut*sin_psi)*(beta - np.arctan((Wa+Wa)/(Wt+Wt))))/(2.*(f_wt_2 + f_wa_2)**(0.5))
                           + (pi*U*(f_wt_2 +f_wa_2)**(0.5)*(U + Utcospsi  +  Uasinpsi))/(2.*(f_wa_2/(f_wt_2) + 1.)*utpUcospsi2)
                           - (4.*U*piece*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5)*(R - r)*(Ut/2. - 
                            (Ucospsi)/2.)*(U + Utcospsi + Uasinpsi ))/(f_wa_2*(1. - np.exp(-(B*(Wt+Wt)*(R - 
                            r))/(r*(Wa+Wa))))**(0.5)) + (128.*U*r*arccos_piece*(Wa+Wa)*(Ut/2. - (Ucospsi)/2.)*(U + 
                            Utcospsi  + Uasinpsi ))/(BBB*pi2*utpUcospsi*utpUcospsi2*((16.*f_wa_2)/(BB*pi2*f_wt_2) + 1.)**(0.5))) 

        dR_dpsi[np.isnan(dR_dpsi)] = 0.1

        dpsi        = -Rsquiggly/dR_dpsi
        PSI         = PSI + dpsi
        diff        = np.max(abs(PSIold-PSI))
        PSIold      = PSI

        # omega = 0, do not run BEMT convergence loop 
        if all(omega[:,0]) == 0. :
            break

        # If its really not going to converge
        if np.any(PSI>pi/2) and np.any(dpsi>0.0):
            break

        ii+=1 
        if ii>10000:
            break

    # More Cd scaling from Mach from AA241ab notes for turbulent skin friction
    Tw_Tinf     = 1. + 1.78*(Ma*Ma)
    Tp_Tinf     = 1. + 0.035*(Ma*Ma) + 0.45*(Tw_Tinf-1.)
    Tp          = (Tp_Tinf)*T
    Rp_Rinf     = (Tp_Tinf**2.5)*(Tp+110.4)/(T+110.4) 
    Cd          = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval  

    epsilon                  = Cd/Cl
    epsilon[epsilon==np.inf] = 10. 

    blade_T_distribution    = rho*(Gamma*(Wt-epsilon*Wa))*deltar 
    blade_Q_distribution    = rho*(Gamma*(Wa+epsilon*Wt)*r)*deltar 
    thrust                  = np.atleast_2d((B * np.sum(blade_T_distribution, axis = 1))).T 
    torque                  = np.atleast_2d((B * np.sum(blade_Q_distribution, axis = 1))).T         
    power                   = omega*torque   

    # calculate coefficients with UIUC definition 
    D        = 2*R 
    Cq       = torque/(rho_0*(n*n)*(D*D*D*D*D)) 
    Ct       = thrust/(rho_0*(n*n)*(D*D*D*D))
    Cp       = power/(rho_0*(n*n*n)*(D*D*D*D*D))
    etap     = V*thrust/power  
     
    Area   = np.pi*(R**2)
    FM  = (thrust*np.sqrt(thrust/2*rho*Area))/power
    FM_constraint    = (FM[0][0] - design_FM)*10
    
    return FM_constraint

def linear_discretize(x_pivs,chi,pivot_points):
    
    chi_pivs       = np.zeros(len(x_pivs))
    chi_pivs[0]    = chi[0]
    chi_pivs[-1]   = chi[-1]
    locations      = np.array(pivot_points)*(chi[-1]-chi[0]) + chi[0]
    
    # vectorize 
    chi_2d = np.repeat(np.atleast_2d(chi).T,len(pivot_points),axis = 1)
    pp_2d  = np.repeat(np.atleast_2d(locations),len(chi),axis = 0)
    idxs   = (np.abs(chi_2d - pp_2d)).argmin(axis = 0) 
    
    chi_pivs[1:-1] = chi[idxs]
    
    x_bar  = np.interp(chi,chi_pivs, x_pivs) 
    
    return x_bar 


def spline_discretize(x_pivs,chi,pivot_points):
    chi_pivs       = np.zeros(len(x_pivs))
    chi_pivs[0]    = chi[0]
    chi_pivs[-1]   = chi[-1]
    locations      = np.array(pivot_points)*(chi[-1]-chi[0]) + chi[0]
    
    # vectorize 
    chi_2d = np.repeat(np.atleast_2d(chi).T,len(pivot_points),axis = 1)
    pp_2d  = np.repeat(np.atleast_2d(locations),len(chi),axis = 0)
    idxs   = (np.abs(chi_2d - pp_2d)).argmin(axis = 0) 
    
    chi_pivs[1:-1] = chi[idxs]
    
    fspline = interpolate.CubicSpline(chi_pivs,x_pivs)
    x_bar = fspline(chi)
    
    return x_bar