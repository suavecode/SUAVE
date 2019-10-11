## @ingroup Components-Energy-Converters
# Propeller.py
#
# Created:  Jun 2014, E. Botero
# Modified: Jan 2016, T. MacDonald
#           Feb 2019, M. Vegh            

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Core import Data
from SUAVE.Methods.Aerodynamics.XFOIL.compute_airfoil_polars import compute_airfoil_polars
from SUAVE.Methods.Geometry.Three_Dimensional \
     import angles_to_dcms, orientation_product, orientation_transpose

# package imports
import numpy as np
import scipy as sp
import scipy.optimize as opt
from scipy.optimize import fsolve

from warnings import warn

# ----------------------------------------------------------------------
#  Propeller Class
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Converters
class Propeller(Energy_Component):
    """This is a propeller component.
    
    Assumptions:
    None
    Source:
    None
    """     
    def __defaults__(self):
        """This sets the default values for the component to function.
        Assumptions:
        None
        Source:
        N/A
        Inputs:
        None
        Outputs:
        None
        Properties Used:
        None
        """         
        self.number_blades            = 0.0
        self.tip_radius               = 0.0
        self.hub_radius               = 0.0
        self.twist_distribution       = 0.0
        self.chord_distribution       = 0.0
        self.mid_chord_aligment       = 0.0
        self.thrust_angle             = 0.0
        self.induced_hover_velocity   = None
        self.airfoil_sections         = None
        self.airfoil_section_location = None
        self.radius_distribution      = None
        self.rotation                 = None
        self.ducted                   = False
        self.induced_power_factor     = 1.48  #accounts for interference effects
        self.profile_drag_coefficient = .03        
        self.tag                      = 'Propeller'
        
    def spin(self,conditions):
        """Analyzes a propeller given geometry and operating conditions.
        
        Assumptions:
        per source
        
        Source:
        Qprop theory document
        
        Inputs:
        self.inputs.omega            [radian/s]
        conditions.freestream.
          density                    [kg/m^3]
          dynamic_viscosity          [kg/(m-s)]
          speed_of_sound             [m/s]
          temperature                [K]
        conditions.frames.
          body.transform_to_inertial (rotation matrix)
          inertial.velocity_vector   [m/s]
        conditions.propulsion.
          throttle                   [-]
          
        Outputs:
        conditions.propulsion.acoustic_outputs.
          number_sections            [-]
          r0                         [m]
          airfoil_chord              [m]
          blades_number              [-]
          propeller_diameter         [m]
          drag_coefficient           [-]
          lift_coefficient           [-]
          omega                      [radian/s]
          velocity                   [m/s]
          thrust                     [N]
          power                      [W]
          mid_chord_aligment         [m] (distance from the mid chord to the line axis out of the center of the blade)
        conditions.propulsion.etap   [-]
        thrust                       [N]
        torque                       [Nm]
        power                        [W]
        Cp                           [-] (coefficient of power)
        Properties Used:
        self. 
          number_blades              [-]
          tip_radius                 [m]
          hub_radius                 [m]
          twist_distribution         [radians]
          chord_distribution         [m]
          mid_chord_aligment         [m] (distance from the mid chord to the line axis out of the center of the blade)
          thrust_angle               [radians]
        """         
        #Unpack            
        B                 = self.number_blades
        R                 = self.tip_radius
        Rh                = self.hub_radius
        local_blade_pitch = self.twist_distribution
        c                 = self.chord_distribution
        omega1            = self.inputs.omega 
        a_sec             = self.airfoil_sections        
        a_secl            = self.airfoil_section_location 
        rho               = conditions.freestream.density[:,0,None]
        mu                = conditions.freestream.dynamic_viscosity[:,0,None]
        Vv                = conditions.frames.inertial.velocity_vector
        Vh                = self.induced_hover_velocity 
        a                 = conditions.freestream.speed_of_sound[:,0,None]
        T                 = conditions.freestream.temperature[:,0,None]
        theta             = self.thrust_angle
        tc                = .12 # Thickness to chord
        ducted            = self.ducted  
        BB                = B*B
        BBB               = BB*B
        
        try:
            pitch_command = conditions.propulsion.pitch_command
            total_blade_pitch = local_blade_pitch + pitch_command   
        except:
            total_blade_pitch = local_blade_pitch 
        
        # Velocity in the Body frame
        T_body2inertial = conditions.frames.body.transform_to_inertial
        T_inertial2body = orientation_transpose(T_body2inertial)
        V_body = orientation_product(T_inertial2body,Vv)
         
        # Velocity transformed to the propulsor frame with flag for tilt rotor
        if np.isscalar(theta):
            body2thrust   = np.array([[np.cos(theta), 0., np.sin(theta)],[0., 1., 0.], [-np.sin(theta), 0., np.cos(theta)]])
            T_body2thrust = orientation_transpose(np.ones_like(T_body2inertial[:])*body2thrust)
        else:
            body2thrust = np.zeros((len(theta),3,3))
            for i in range(len(theta)):
                body2thrust[i,:,:] = [[np.cos(theta[i][0]), 0., np.sin(theta[i][0])],[0., 1., 0.], [-np.sin(theta[i][0]), 0., np.cos(theta[i][0])]]
            T_body2thrust = orientation_transpose(body2thrust)
    
        V_thrust = orientation_product(T_body2thrust,V_body)
    
        # Now just use the aligned velocity
        V = V_thrust[:,0,None]
        V_inf = V_thrust 
        
        ua = np.zeros_like(V)
        if Vh != None:     
            for i in range(len(V)): 
                V_Vh =  V_thrust[i][0]/Vh
                if Vv[i,:].all()  == True :
                    ua[i] = Vh
                elif Vv[i][0]  == 0 and  Vv[i][2] != 0: # vertical / axial flight
                    if V_Vh > 0: # climbing 
                        ua[i] = Vh*(-(-V_inf[i][0]/(2*Vh)) + np.sqrt((-V_inf[i][0]/(2*Vh))**2 + 1))
                    elif -2 <= V_Vh and V_Vh <= 0:  # slow descent                 
                        ua[i] = Vh*(1.15 -1.125*(V_Vh) - 1.372*(V_Vh)**2 - 1.718*(V_Vh)**2 - 0.655*(V_Vh)**4 ) 
                    else: # windmilling 
                        print("rotor is in the windmill break state!")
                        ua[i] = Vh*(-(-V_inf[i][0]/(2*Vh)) - np.sqrt((-V_inf[i][0]/(2*Vh))**2 + 1))
                else: # forward flight conditions                 
                    func = lambda vi: vi - (Vh**2)/(np.sqrt(((-V_inf[i][2])**2 + (V_inf[i][0] + vi)**2)))
                    vi_initial_guess = V_inf[i][0]
                    ua[i]    = fsolve(func,vi_initial_guess)
            lamda_i      = ua/(omega*R)
        else:              
            ut       = 0.0  
         
        #Things that don't change with iteration
        N        = len(c) # Number of stations     
        ctrl_pts = len(Vv) 
        
        # if airfoils are specified, get CL and CD as a function of alpha (currently not used)
        if  a_sec != None and a_secl != None:
            airfoil_polars = Data()
            # check dimension of section  
            dim_sec = len(a_secl)
            if dim_sec != N:
                raise AssertionError("Number of sections not equal to number of stations")
            # compute airfoil polars for airfoils 
            airfoil_polars = compute_airfoil_polars(self,conditions, a_sec)
            airfoil_cl     = airfoil_polars.CL
            airfoil_cd     = airfoil_polars.CD
            AoA_range      = airfoil_polars.AoA_range
        
        # set up non dimensional radial distribution 
        if self.radius_distribution is None:
            chi0    = Rh/R   # Where the propeller blade actually starts
            chi     = np.linspace(chi0,1,N+1)  # Vector of nondimensional radii
            chi     = chi[0:N]
        
        else:
            chi = self.radius_distribution
            
        omega = omega1*1.0
        omega = np.abs(omega)        
        r_dim = chi*R                        # Radial coordinate 
        pi    = np.pi
        pi2   = pi*pi   
        A     = pi*(R**2)
        x     = r_dim*np.multiply(omega,1/V) # Nondimensional distance
        n     = omega/(2.*pi)                # Cycles per second
        J     = V/(2.*R*n)     
        
        # blade area 
        blade_area   = sp.integrate.cumtrapz(B*c, r_dim-r_dim[0])
        
        # solidity 
        sigma        = blade_area[-1]/(pi*r_dim[-1]**2)   
        
        # compute lambda and mu 
        lamda_mean   = (np.atleast_2d(V_inf[:,0]).T + ua)/(omega*R)
        
        mu_prop      = (np.atleast_2d(V_inf[:,2]).T) /(omega*R)  
        alpha_disc   = np.arctan(np.atleast_2d(V_inf[:,0]).T/V_inf[:,2])
        lamda_c      = (np.atleast_2d(V_inf[:,0]).T)/(omega*R)
        
        # wake skew angle 
        X            = np.arctan(mu_prop/lamda_mean)
        kx           = np.tan(X/2)
        
        # blade flap rate and sweep(cone) angle 
        beta_dot        = 0  # currently no flaping 
        beta            = 0  # currently no coning         
        
        # azimuth distribution 
        psi          = np.linspace(0,2*pi,N)
        psi_2d       = np.tile(np.atleast_2d(psi).T,(1,N))
        psi_2d       = np.repeat(psi_2d[np.newaxis, :, :], ctrl_pts, axis=0)  
        
        # 2 dimensiona radial distribution 
        chi_2d       = np.tile(chi ,(N,1))            
        r_2d         = np.repeat(chi_2d[ np.newaxis,:, :], ctrl_pts, axis=0) 
        
        # Momentum theory approximation of inflow for BET if the advance ratio is large
        mu_lamda = lamda_c/abs(mu_prop) 
        if any(mu_lamda[:,0] < 10.0):
            
            # create radial distribution and aximuthal distribution  
            theta_2d        = np.tile(total_blade_pitch,(N ,1))
            theta_blade_2d  = np.repeat(theta_2d[ np.newaxis,:, :], ctrl_pts, axis=0)    
            omega_2d        = np.tile(np.atleast_3d(omega),(1,N,N))
            a_2d            = np.tile(np.atleast_3d(a),(1,N,N))
            mu_2d           = np.tile(np.atleast_3d(mu),(1,N,N))
            rho_2d          = np.tile(np.atleast_3d(rho),(1,N,N))
            T_2d            = np.tile(np.atleast_3d(T),(1,N,N))
            lamda_c         = np.tile(np.atleast_3d(lamda_c),(1,N,N))
            lamda_mean      = np.tile(np.atleast_3d(lamda_mean),(1,N,N))
            mu_prop         = np.tile(np.atleast_3d(mu_prop),(1,N,N))  
            
            beta_tip        = np.tile(np.atleast_3d(theta_2d[:,-1]),(ctrl_pts,1,N))
            tip_loss_factor = (2/pi)*np.arccos(np.exp(-B*(1-r_2d)/(2*np.sin(beta_tip))))   
            
            # wake skew angle 
            X  = np.arctan(mu_prop/lamda_mean)
            kx = np.tan(X/2) 
     
            # initial radial inflow distribution 
            lamda_i = lamda_mean*(1 + kx*r_2d*np.cos(psi_2d))     
            lamda   = lamda_c + lamda_i
            
            # axial, tangential and radial components of local blade flow 
            y  = r_2d*R
            ut = omega_2d*y + mu_prop*omega_2d*R*np.sin(psi_2d)                  
            ur = mu_prop*omega_2d*R*np.cos(psi_2d)                                      
            up = lamda*omega_2d*R  + r_2d*beta_dot + mu_prop*omega_2d*R*beta*np.cos(psi_2d)  
            
            # Total speed at blade 
            U   = np.sqrt(ut**2 + ur**2 + up**2 )
            
            # local Mach number at blade 
            Ma  = U/a_2d  
            
            # blade incident angle 
            phi = np.arctan(up/ut)
            
            # local blade angle of attack
            alpha  = theta_blade_2d - phi 
            
            # Estimate Cl max
            nu_2d         =  mu_2d/rho_2d
            Re         = (U*c)/nu_2d 
            Cl_max_ref = -0.0009*tc**3 + 0.0217*tc**2 - 0.0442*tc + 0.7005
            Re_ref     = 9.*10**6      
            Cl1maxp    = Cl_max_ref * ( Re / Re_ref ) **0.1
            
            # Ok, from the airfoil data, given Re, Ma, alpha we need to find Cl
            Cl = 2.*pi*alpha
            
            # By 90 deg, it's totally stalled.
            Cl[Cl>Cl1maxp]  = Cl1maxp[Cl>Cl1maxp]  
            Cl[alpha>=pi/2] = 0.
                
            # Scale for Mach, this is Karmen_Tsien
            Cl[Ma[:,:]<1.] = Cl[Ma[:,:]<1.]/((1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5+((Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])/(1+(1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5))*Cl[Ma<1.]/2)
        
            # If the blade segments are supersonic, don't scale
            Cl[Ma[:,:]>=1.] = Cl[Ma[:,:]>=1.]     
            
            #There is also RE scaling
            #This is an atrocious fit of DAE51 data at RE=50k for Cd
            Cdval = (0.108*(Cl*Cl*Cl*Cl)-0.2612*(Cl*Cl*Cl)+0.181*(Cl*Cl)-0.0139*Cl+0.0278)*((50000./Re)**0.2)
            Cdval[alpha>=pi/2] = 2.
            
            #More Cd scaling from Mach from AA241ab notes for turbulent skin friction
            Tw_Tinf = 1. + 1.78*(Ma*Ma)
            Tp_Tinf = 1. + 0.035*(Ma*Ma) + 0.45*(Tw_Tinf-1.)
            Tp      = (Tp_Tinf)*T_2d
            Rp_Rinf = (Tp_Tinf**2.5)*(Tp+110.4)/(T_2d+110.4)
            
            Cd = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval 
            
            # total velocity 
            U = np.sqrt(ut**2 + up**2)
            
            # local blade lift and drag 
            dL = 0.5 * rho_2d * U**2 * c * Cl
            dD = 0.5 * rho_2d * U**2 * c * Cd
            
            # application of tip loss factor 
            dL[r_2d>0.97] = 0 
            
            # normal and tangential forces  
            dFz = dL*np.cos(phi) - dD*np.sin(phi)  
            dFx = dL*np.sin(phi) - dD*np.cos(phi) 
            
            # average thrust and torque over aximuth
            deltar               = np.ones_like(dFz)*(chi[1]-chi[0])   
            blade_T_distribution = np.mean((dFz*deltar), axis = 1)
            blade_Q_distribution = np.mean((dFx*chi*deltar), axis = 1)
            thrust               = np.atleast_2d((B * np.sum(blade_T_distribution, axis = 1))).T 
            torque               = np.atleast_2d((B * np.sum(blade_Q_distribution, axis = 1))).T         
    
            va_2d = up    
            vt_2d = ut 
            
        # Blade Element Momentum Theory : large angle approximation
        else:  
            # radial distribution 
            r = np.tile(chi,(ctrl_pts,1))  
            
            # blade pitch distribution            
            theta_blade  = np.tile(total_blade_pitch,(ctrl_pts,1)) 
            
            # chord distribution 
            local_chord  = np.tile(c,(ctrl_pts,1))   
            
            # freestream inflow ratio 
            lamda_c  = np.ones_like(local_chord)*lamda_c  
            
            # initial guess for induced inflow ratio
            lamda_i_old  = np.ones_like(local_chord)*0.1  
            
            # intial guess for total inflow ratio
            lamda  = np.tile(np.atleast_2d(lamda_mean),(1 ,N))  
            
            # Setup a Newton iteration 	  
            tol    = 1e-5
            ii     = 0  	        
            broke  = False      	
            diff   = 1.	 
            
            while (diff > tol):                    
                # axial, tangential and radial components of local blade flow 	   
                ut = omega*r*R                       
                up = lamda*omega*R 
                
                # total speed at blade 
                U = np.sqrt(ut**2 + up**2)
                
                # local Mach number at blade 
                Ma = U/a  
        
                # blade incident angle 	
                phi = np.arctan(up/ut)
        
                # local blade angle of attact 
                alpha = theta_blade - phi   
                
                phi_tip = np.tile(np.atleast_2d(phi[:,-1]).T  ,(1 ,N))      
                tip_loss_factor = (2/pi)*np.arccos(np.exp(-B *(1-r)/(2*np.sin(phi_tip)))) 
        
                # Estimate Cl max
                nu         = mu/rho
                Re         = (U*local_chord )/nu 
                Cl_max_ref = -0.0009*tc**3 + 0.0217*tc**2 - 0.0442*tc + 0.7005
                Re_ref     = 9.*10**6      
                Cl1maxp    = Cl_max_ref * ( Re / Re_ref ) **0.1
        
                # Ok, from the airfoil data, given Re, Ma, alpha we need to find Cl
                Cl = 2.*pi*alpha
        
                # By 90 deg, it's totally stalled.
                Cl[Cl>Cl1maxp]  = Cl1maxp[Cl>Cl1maxp]  
                Cl[alpha>=pi/2] = 0.
        
                # Scale for Mach, this is Karmen_Tsien
                Cl[Ma[:,:]<1.] = Cl[Ma[:,:]<1.]/((1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5+((Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])/(1+(1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5))*Cl[Ma<1.]/2)
        
                # If the blade segments are supersonic, don't scale
                Cl[Ma[:,:]>=1.] = Cl[Ma[:,:]>=1.]     
        
                #There is also RE scaling
                #This is an atrocious fit of DAE51 data at RE=50k for Cd
                Cdval = (0.108*(Cl*Cl*Cl*Cl)-0.2612*(Cl*Cl*Cl)+0.181*(Cl*Cl)-0.0139*Cl+0.0278)*((50000./Re)**0.2)
                Cdval[alpha>=pi/2] = 2.
        
                #More Cd scaling from Mach from AA241ab notes for turbulent skin friction
                Tw_Tinf = 1. + 1.78*(Ma*Ma)
                Tp_Tinf = 1. + 0.035*(Ma*Ma) + 0.45*(Tw_Tinf-1.)
                Tp      = (Tp_Tinf)*T
                Rp_Rinf = (Tp_Tinf**2.5)*(Tp+110.4)/(T+110.4)
        
                Cd = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval 	
        
                # force coefficient 	
                cFz             = Cl*np.cos(phi) - Cd *np.sin(phi)	
        
                # newtown raphson iteration 	 
                a1              = 1 - (sigma*cFz)/(8*r*tip_loss_factor)
                f_lambda_i      = (lamda_i_old**2)*a1 + lamda_i_old*lamda_c*((2*a1)-1) + (a1-1)*(r**2 + lamda_c**2)
                f_prime_lamda_i = 2*(lamda_i_old)*a1  + lamda_c*((2*a1)-1) 
                lamda_i_next    = lamda_i_old - f_lambda_i/f_prime_lamda_i 	
                relaxation      = 0.95
                lamda_i_new     = lamda_i_old*relaxation +  lamda_i_next*(1-relaxation)
        
                # get difference of old and new solution for lambda 	
                diff            = np.max(abs(lamda_i_new - lamda_i_old))
        
                # in the event that the tolerance is not met	
                # a) make recently calulated value the new value for next iteration 	
                lamda_i_old     = lamda_i_new 	 
        
                lamda   = lamda_i_new + lamda_c	                
        
                ii+=1 	
                if ii>5000:	
                    # maximum iterations is 2000	
                    broke = True	
                    break
                
            # local blade lift and drag 
            dL   = 0.5 * rho * U**2 * local_chord * Cl
            dD   = 0.5 * rho * U**2 * local_chord * Cd
                
            # normal and tangential forces 
            dFz  = dL*np.cos(phi) - dD*np.sin(phi)  
            dFx  = dL*np.sin(phi) - dD*np.cos(phi) 
                
            # average thrust and torque over aximuth
            deltar               = np.tile(np.atleast_2d((r[:,1]-r[:,0])).T  ,(1 ,N))    
            blade_T_distribution = dFz*deltar
            blade_Q_distribution = dFx*r*deltar
            thrust               = np.atleast_2d(B * np.sum(blade_T_distribution,  axis = 1 )).T
            torque               = np.atleast_2d(B * np.sum(blade_Q_distribution,  axis = 1 )).T
              
            va_2d = np.repeat(up.T[ np.newaxis,:  , :], N, axis=0) 
            vt_2d = np.repeat(ut.T[ np.newaxis,:  , :], N, axis=0)  
                 
        
        D        = 2*R 
        Cq       = torque/(rho*A*R*(omega*R)**2)
        Ct       = thrust/(rho*A*(omega*R)**2)
        Ct[Ct<0] = 0.     # prevent things from breaking
        kappa    = self.induced_power_factor 
        Cd0      = self.profile_drag_coefficient   
        Cp       = np.zeros_like(Ct)
        power    = np.zeros_like(Ct)        
        blade_dT_dR = np.zeros_like(deltar)
        blade_dT_dr = np.zeros_like(deltar)
        blade_dQ_dR = np.zeros_like(deltar)
        blade_dQ_dr = np.zeros_like(deltar)
        
        for i in range(len(Vv)):
            blade_dT_dR[i,:] = np.gradient(blade_T_distribution[i], deltar[i,0]*R) 
            blade_dT_dr[i,:] = np.gradient(blade_T_distribution[i], deltar[i,0])
            blade_dQ_dR[i,:] = np.gradient(blade_Q_distribution[i], deltar[i,0]*R)
            blade_dQ_dr[i,:] = np.gradient(blade_Q_distribution[i], deltar[i,0])  
            if -1. <Vv[i][0] <1.: # vertical/axial flight
                Cp[i]       = (kappa*(Ct[i]**1.5)/(2**.5))+sigma*Cd0/8.
                power[i]    = Cp[i]*(rho[i]*(n[i]*n[i]*n[i])*(D*D*D*D*D))
                torque[i]   = power[i]/omega[i]  
            else:  
                power[i]    = torque[i]*omega[i]   
                Cp[i]       = power[i]/(rho[i]*(n[i]*n[i]*n[i])*(D*D*D*D*D))
 
        etap     = V*thrust/power     
        
        conditions.propulsion.etap = etap
        
        # store data
        results_conditions = Data     
        outputs   = results_conditions(
            n_blades                  = B,
            R                         = R,
            D                         = D,
            number_sections           = N,
            radius_distribution       = np.linspace(Rh ,R, N),
            chord_distribution        = c,     
            twist_distribution        = total_blade_pitch,            
            r_2d                      = r_2d,
            r                         = chi,
            thrust_angle              = theta,
            speed_of_sound            = conditions.freestream.speed_of_sound,
            density                   = conditions.freestream.density,
            velocity                  = Vv, 
            vt                        = vt_2d, 
            va                        = va_2d, 
            drag_coefficient          = Cd,
            lift_coefficient          = Cl,       
            omega                     = omega, 
            blade_dT_dR               = blade_dT_dR,
            blade_dT_dr               = blade_dT_dr,
            blade_T_distribution      = blade_T_distribution, 
            blade_T                   = thrust/B,  
            Ct                        = Ct, 
            psi                       = psi,            
            blade_dQ_dR               = blade_dQ_dR ,
            blade_dQ_dr               = blade_dQ_dr ,
            blade_Q_distribution      = blade_Q_distribution ,
            blade_Q                   = torque/B,   
            Cq                        = Cq,             
            power                     = power,            
            mid_chord_aligment        = self.mid_chord_aligment     
        ) 
        
        return thrust, torque, power, Cp , outputs  , etap  

    
    def spin_surrogate(self,conditions):
        
        # unpack
        surrogate = self.surrogate
        altitude  = conditions.freestream.altitude
        Vv        = conditions.frames.inertial.velocity_vector
        rho       = conditions.freestream.density[:,0,None]        
        omega     = self.inputs.omega
        R         = self.tip_radius 
        theta     = self.thrust_angle
        
        
        # Velocity in the Body frame
        T_body2inertial = conditions.frames.body.transform_to_inertial
        T_inertial2body = orientation_transpose(T_body2inertial)
        V_body = orientation_product(T_inertial2body,Vv)
    
        # Velocity transformed to the propulsor frame
        body2thrust   = np.array([[np.cos(theta), 0., np.sin(theta)],[0., 1., 0.], [-np.sin(theta), 0., np.cos(theta)]])
        T_body2thrust = orientation_transpose(np.ones_like(T_body2inertial[:])*body2thrust)
        V_thrust      = orientation_product(T_body2thrust,V_body)
    
        # Now just use the aligned velocity
        velocity = V_thrust[:,0,None] 
        
        velocity[velocity==0.] = np.sqrt(self.design_thrust/(2*rho[velocity==0.]*np.pi*(self.tip_radius**2)))

        # Diameter
        D = R*2  
        
        omega[omega==0] = 1e-6 

        # Advance Ratio
        n = omega/(2*np.pi)
        J = velocity/(n*D)
        
        # Surrogate input
        xyz = np.hstack([J,altitude])
        
        # Use the surrogate
        eta = surrogate.efficiency.predict(xyz)
        Cp  = surrogate.power_coefficient.predict(xyz)
        
        # Get results
        Ct  = eta*Cp/J
        Cq  = Cp/(2*np.pi)
        
        thrust = Ct*rho*(n**2)*(D**4)
        torque = Cq*rho*(n**2)*(D**5)
        power  = Cp*rho*(n**3)*(D**5)
        
        #thrust[omega<0.0] = - thrust[omega<0.0]
        
        conditions.propulsion.etap = eta

        return thrust, torque, power, Cp  