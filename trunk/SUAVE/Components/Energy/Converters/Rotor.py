## @ingroup Components-Energy-Converters
# Rotor.py
#
# Created:  Jun 2014, E. Botero
# Modified: Jan 2016, T. MacDonald
#           Feb 2019, M. Vegh            
#           Mar 2020, M. Clarke
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
import scipy as sp
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Core import Data, Units
import scipy.optimize as opt
from scipy.optimize import fsolve
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_polars import compute_airfoil_polars
from SUAVE.Methods.Geometry.Three_Dimensional \
     import angles_to_dcms, orientation_product, orientation_transpose

from warnings import warn

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ----------------------------------------------------------------------
#  Rotor Class
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Converters
class Rotor(Energy_Component):
    """This is a rotor component.
    
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
        self.blade_solidity           = 0.0
        self.thrust_angle             = 0.0 
        self.design_power             = None
        self.design_thrust            = None           
        self.induced_hover_velocity   = None
        self.airfoil_geometry         = None
        self.airfoil_polars           = None
        self.airfoil_polar_stations   = None 
        self.radius_distribution      = None
        self.rotation                 = None
        self.ducted                   = False
        self.induced_power_factor     = 1.15  #accounts for interference effeces
        self.profile_drag_coefficient = .01
        self.lift_curve_slope         = 2*np.pi
        self.tag                      = 'Rotor'
        
    def spin(self,conditions):
        """Analyzes a rotor given geometry and operating conditions.
        
        Assumptions:
        per source
        
        Source:
        Drela, M. "Qprop Formulation", MIT AeroAstro, June 2006
        http://web.mit.edu/drela/Public/web/qprop/qprop_theory.pdf
        
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
          rotor_diameter         [m]
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
        B      = self.number_blades
        R      = self.tip_radius
        Rh     = self.hub_radius
        beta   = self.twist_distribution
        c      = self.chord_distribution
        chi    = self.radius_distribution
        omega  = self.inputs.omega 
        a_geo  = self.airfoil_geometry
        a_pol  = self.airfoil_polars        
        a_loc  = self.airfoil_polar_stations             
        rho    = conditions.freestream.density[:,0,None]
        mu     = conditions.freestream.dynamic_viscosity[:,0,None]
        Vv     = conditions.frames.inertial.velocity_vector
        Vh     = self.induced_hover_velocity 
        a      = conditions.freestream.speed_of_sound[:,0,None]
        T      = conditions.freestream.temperature[:,0,None]
        theta  = self.thrust_angle
        tc     = self.thickness_to_chord
        sigma  = self.blade_solidity     
        
        BB        = B*B
        BBB       = BB*B
        disk_area = np.pi*(R**2)
        kappa     = self.induced_power_factor 
        
        # Velocity in the Body frame
        T_body2inertial = conditions.frames.body.transform_to_inertial
        T_inertial2body = orientation_transpose(T_body2inertial)
        V_body          = orientation_product(T_inertial2body,Vv)
        
        # Velocity in the Body frame
        T_body2inertial = conditions.frames.body.transform_to_inertial
        T_inertial2body = orientation_transpose(T_body2inertial)
        V_body          = orientation_product(T_inertial2body,Vv)
        body2thrust     = np.array([[np.cos(theta), 0., np.sin(theta)],[0., 1., 0.], [-np.sin(theta), 0., np.cos(theta)]])
        T_body2thrust   = orientation_transpose(np.ones_like(T_body2inertial[:])*body2thrust)  
        V_thrust        = orientation_product(T_body2thrust,V_body)
        
        # Now just use the aligned velocity
        V = V_thrust[:,0,None]
        
        ua = np.zeros_like(V)
        power_ratio = np.zeros_like(V)
        if Vh != None:   
            for i in range(len(V)):
                V_inf = V_thrust[i] 
                V_Vh =  V_thrust[i][0]/Vh
            
                if Vv[i,:].all()  == 0 :
                    ua[i] = Vh
                elif Vv[i][0]  == 0 and  Vv[i][2] != 0: # vertical / axial flight
                    if V_Vh > 0: # climbing 
                        ua[i] =Vh*(-.5*V_Vh+np.sqrt((.5*V_Vh)**2+1)) #Vh*(-(-V_inf[0]/(2*Vh)) + np.sqrt((-V_inf[0]/(2*Vh))**2 + 1))
                        
                    elif -2 <= V_Vh and V_Vh <= 0:  # slow descent                 
                        ua[i] = Vh*(1.15-V_Vh)#Vh*(1.15 -1.125*(V_Vh) - 1.372*(V_Vh)**2 - 1.718*(V_Vh)**2 - 0.655*(V_Vh)**4 ) 
                    else: # windmilling 
                        print("rotor is in the windmill break state!")
                        ua[i] = Vh*(-(-V_inf[0]/(2*Vh)) - np.sqrt((-V_inf[0]/(2*Vh))**2 + 1))
                
                
                else: # forward flight conditions                 
                    func = lambda vi: vi - (Vh**2)/(np.sqrt(((-V_inf[2])**2 + (V_inf[0] + vi)**2)))
                    vi_initial_guess = V_inf[0]
                    ua[i]    = fsolve(func,vi_initial_guess)
           
                power_ratio[i] = ua[i]/Vh+V_Vh 
        else: 
            ua = 0.0 
        
        ut = 0.0
        
        nu    = mu/rho
        tol   = 1e-5 # Convergence tolerance 
        
        #Things that don't change with iteration
        N       = len(c) # Number of stations     
        
        if  a_pol != None and a_loc != None:
            airfoil_polars = Data() 
            # check dimension of section
            if len(a_loc) != N:
                raise AssertionError('Dimension of airfoil sections must be equal to number of stations on rotor')
            # compute airfoil polars for airfoils 
            airfoil_polars = compute_airfoil_polars(self, a_geo, a_pol)
            airfoil_cl     = airfoil_polars.lift_coefficients
            airfoil_cd     = airfoil_polars.drag_coefficients
            AoA_sweep      = airfoil_polars.angle_of_attacks
        
        if self.radius_distribution is None:
            chi0    = Rh/R   # Where the rotor blade actually starts
            chi     = np.linspace(chi0,1,N+1)  # Vector of nondimensional radii
            chi     = chi[0:N] 
        
        lamda       = V/(omega*R)              # Speed ratio
        r           = chi*R                    # Radial coordinate
        pi          = np.pi
        pi2         = pi*pi
        x           = r*np.multiply(omega,1/V) # Nondimensional distance
        n           = omega/(2.*pi)            # Cycles per second
        J           = V/(2.*R*n)     
        blade_area  = sp.integrate.cumtrapz(B*c, r-r[0])
        sigma       = blade_area[-1]/(pi*r[-1]**2)
        omegar      = np.outer(omega,r)
        Ua          = np.outer((V + ua),np.ones_like(r))
        Ut          = omegar - ut
        U           = np.sqrt(Ua*Ua + Ut*Ut)
        
        #Things that will change with iteration
        size = (len(a),N)
        Cl = np.zeros((1,N)) 
        
        #Setup a Newton iteration
        psi    = np.ones(size)
        psiold = np.zeros(size)
        diff   = 1.
        
        ii = 0
        broke = False        
        while (diff>tol):
            sin_psi = np.sin(psi)
            cos_psi = np.cos(psi)
            Wa      = 0.5*Ua + 0.5*U*sin_psi
            Wt      = 0.5*Ut + 0.5*U*cos_psi   
            va      = Wa - Ua
            vt      = Ut - Wt
            alpha   = beta - np.arctan2(Wa,Wt)
            W       = (Wa*Wa + Wt*Wt)**0.5
            Ma      = (W)/a #a is the speed of sound  
            lamdaw  = r*Wa/(R*Wt)
            
            # Limiter to keep from Nan-ing
            lamdaw[lamdaw<0.] = 0.
            
            f            = (B/2.)*(1.-r/R)/lamdaw
            piece        = np.exp(-f)
            arccos_piece = np.arccos(piece)
            F            = 2.*arccos_piece/pi
            Gamma        = vt*(4.*pi*r/B)*F*(1.+(4.*lamdaw*R/(pi*B*r))*(4.*lamdaw*R/(pi*B*r)))**0.5
            
            # Estimate Cl max
            Re         = (W*c)/nu 
            Cl_max_ref = -0.0009*tc**3 + 0.0217*tc**2 - 0.0442*tc + 0.7005
            Re_ref     = 9.*10**6      
            Cl1maxp    = Cl_max_ref * ( Re / Re_ref ) **0.1
            
            # Compute blade CL distribution from the airfoil data 
            if  a_pol != None and a_loc != None: 
                for k in range(N):
                    Cl[0,k] = np.interp(alpha[0,k],AoA_sweep,airfoil_cl[a_loc[k]])
            else:
                # If not airfoil polar provided, use 2*pi as lift curve slope
                Cl = 2.*pi*alpha
            
            # By 90 deg, it's totally stalled.
            Cl[Cl>Cl1maxp]  = Cl1maxp[Cl>Cl1maxp] # This line of code is what changed the regression testing
            Cl[alpha>=pi/2] = 0.
                
            # Scale for Mach, this is Karmen_Tsien
            Cl[Ma[:,:]<1.] = Cl[Ma[:,:]<1.]/((1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5+((Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])/(1+(1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5))*Cl[Ma<1.]/2)
        
            # If the blade segments are supersonic, don't scale
            Cl[Ma[:,:]>=1.] = Cl[Ma[:,:]>=1.] 
        
            Rsquiggly = Gamma - 0.5*W*c*Cl
            
            #An analytical derivative for dR_dpsi, this is derived by taking a derivative of the above equations
            #This was solved symbolically in Matlab and exported        
            f_wt_2 = 4*Wt*Wt
            f_wa_2 = 4*Wa*Wa
            Ucospsi  = U*cos_psi
            Usinpsi  = U*sin_psi
            Utcospsi = Ut*cos_psi
            Uasinpsi = Ua*sin_psi
            
            UapUsinpsi = (Ua + Usinpsi)
            utpUcospsi = (Ut + Ucospsi)
            
            utpUcospsi2 = utpUcospsi*utpUcospsi
            UapUsinpsi2 = UapUsinpsi*UapUsinpsi
            
            dR_dpsi = ((4.*U*r*arccos_piece*sin_psi*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5))/B - 
                       (pi*U*(Ua*cos_psi - Ut*sin_psi)*(beta - np.arctan((Wa+Wa)/(Wt+Wt))))/(2.*(f_wt_2 + f_wa_2)**(0.5))
                       + (pi*U*(f_wt_2 +f_wa_2)**(0.5)*(U + Utcospsi  +  Uasinpsi))/(2.*(f_wa_2/(f_wt_2) + 1.)*utpUcospsi2)
                       - (4.*U*piece*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5)*(R - r)*(Ut/2. - 
                      (Ucospsi)/2.)*(U + Utcospsi + Uasinpsi ))/(f_wa_2*(1. - np.exp(-(B*(Wt+Wt)*(R - 
                       r))/(r*(Wa+Wa))))**(0.5)) + (128.*U*r*arccos_piece*(Wa+Wa)*(Ut/2. - (Ucospsi)/2.)*(U + 
                       Utcospsi  + Uasinpsi ))/(BBB*pi2*utpUcospsi*utpUcospsi2*((16.*f_wa_2)/(BB*pi2*f_wt_2) + 1.)**(0.5))) 
            
            dR_dpsi[np.isnan(dR_dpsi)] = 0.1
                      
            dpsi   = -Rsquiggly/dR_dpsi
            psi    = psi + dpsi
            diff   = np.max(abs(psiold-psi))
            psiold = psi
            
            # If its really not going to converge
            if np.any(psi>(pi*85.0/180.)) and np.any(dpsi>0.0):
                break
                
            ii+=1
                
            if ii>2000:
                broke = True
                break
        
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
        epsilon  = Cd/Cl
        epsilon[epsilon==np.inf] = 10. 
        deltar   = (r[1]-r[0])
        thrust   = rho*B*(np.sum(Gamma*(Wt-epsilon*Wa)*deltar,axis=1)[:,None])
        torque   = rho*B*np.sum(Gamma*(Wa+epsilon*Wt)*r*deltar,axis=1)[:,None]
        D        = 2*R 
        tip_speed = omega*R
        
        #Leishman's thrust coefficient for rotor
        Ctl        = thrust/(rho*disk_area*( tip_speed*tip_speed))   
        Ctl[Ctl<0] = 0. # prevent things from breaking
         
        # motor thrust coefficient
        Ct       = thrust/(rho*(n*n)*(D*D*D*D)) # used for motor model
        Ct[Ct<0] = 0.  
        Cd0      = self.profile_drag_coefficient   
        Cp       = np.zeros_like(Ct)
        Cpl      = np.zeros_like(Ct)
        power    = np.zeros_like(Ct) 
        for i in range(len(Vv)): 
            if -1. < Vv[i][0] < 1.: # vertical/axial flight 
                Cpl[i]      = (kappa*(Ctl[i]**1.5)/(2**.5))+sigma*Cd0/8. # Eqn 2.43 Principles of Helicopter Aerodynamics 
                power[i]    = Cpl[i]*rho[i]*disk_area*(tip_speed[i]*tip_speed[i]*tip_speed[i]) 
                torque[i]   = power[i]/omega[i]    
            else:   
                power[i]    = torque[i]*omega[i]   
                torque[i]   = power[i]/omega[i]  
            Cp[i] = power[i]/(rho[i]*(n[i]*n[i]*n[i])*(D*D*D*D*D))   
        
        # torque coefficient 
        Cq = torque/(rho*(n*n)*(D*D*D*D)*R) 
        
        thrust[conditions.propulsion.throttle[:,0] <=0.0] = 0.0
        power[conditions.propulsion.throttle[:,0]  <=0.0] = 0.0 
        torque[conditions.propulsion.throttle[:,0]  <=0.0] = 0.0 
        thrust[omega<0.0] = - thrust[omega<0.0] 
        
        etap     = V*thrust/power  
        conditions.propulsion.etap = etap
         
        # store data
        results_conditions                   = Data     
        outputs                              = results_conditions(
            num_blades                       = B,
            rotor_radius                     = R,
            rotor_diameter                   = D,
            number_sections                  = N,
            radius_distribution              = np.linspace(Rh ,R, N),
            chord_distribution               = c,     
            twist_distribution               = beta,            
            normalized_radial_distribution   = r,
            thrust_angle                     = theta,
            speed_of_sound                   = conditions.freestream.speed_of_sound,
            density                          = conditions.freestream.density,
            velocity                         = Vv, 
            tangential_velocity_distribution = vt, 
            axial_velocity_distribution      = va, 
            drag_coefficient                 = Cd,
            lift_coefficient                 = Cl,       
            omega                            = omega, 
            dT_dR                            = rho*(Gamma*(Wt-epsilon*Wa)),   
            dT_dr                            = rho*(Gamma*(Wt-epsilon*Wa))*R,  
            thrust_distribution              = rho*(Gamma*(Wt-epsilon*Wa))*deltar, 
            thrust_per_blade                 = thrust/B,  
            thrust_coefficient               = Ct,  
            dQ_dR                            = rho*(Gamma*(Wa+epsilon*Wt)*r), 
            dQ_dr                            = rho*(Gamma*(Wa+epsilon*Wt)*r)*R,
            torque_distribution              = rho*(Gamma*(Wa+epsilon*Wt)*r)*deltar,
            torque_per_blade                 = torque/B,   
            torque_coefficient               = Cq,   
            power                            = power,
            power_coefficient                = Cp, 
            mid_chord_aligment               = self.mid_chord_aligment     
        ) 
 
        return thrust, torque, power, Cp, outputs  , etap  


    

    def spin_variable_pitch(self,conditions):
        """Analyzes a rotor given geometry and operating conditions.

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
          pitch_command              [radian/s] 

        Outputs:
        conditions.propulsion.acoustic_outputs.
          number_sections            [-]
          r0                         [m]
          airfoil_chord              [m]
          blades_number              [-]
          rotor_diameter         [m]
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
        B         = self.number_blades
        R         = self.tip_radius
        Rh        = self.hub_radius        
        beta_in   = self.twist_distribution
        c         = self.chord_distribution
        chi       = self.radius_distribution
        Vh        = self.induced_hover_velocity 
        omega     = self.inputs.omega
        a_geo     = self.airfoil_geometry
        a_pol     = self.airfoil_polars        
        a_loc     = self.airfoil_polar_stations          
        rho       = conditions.freestream.density[:,0,None]
        mu        = conditions.freestream.dynamic_viscosity[:,0,None]
        Vv        = conditions.frames.inertial.velocity_vector
        a         = conditions.freestream.speed_of_sound[:,0,None]
        T         = conditions.freestream.temperature[:,0,None]
        theta     = self.thrust_angle
        tc        = self.thickness_to_chord 
        beta_c    = conditions.propulsion.pitch_command
        sigma     = self.blade_solidity     
        ducted    = self.ducted 
        
        beta      = beta_in + beta_c 
        BB        = B*B
        BBB       = BB*B
        disk_area = np.pi*(R**2)     
        kappa     = self.induced_power_factor 
            
        # Velocity in the Body frame
        T_body2inertial = conditions.frames.body.transform_to_inertial
        T_inertial2body = orientation_transpose(T_body2inertial)
        V_body = orientation_product(T_inertial2body,Vv)
        
        # Velocity in the Body frame
        T_body2inertial = conditions.frames.body.transform_to_inertial
        T_inertial2body = orientation_transpose(T_body2inertial)
        V_body          = orientation_product(T_inertial2body,Vv)
        body2thrust     = np.array([[np.cos(theta), 0., np.sin(theta)],[0., 1., 0.], [-np.sin(theta), 0., np.cos(theta)]])
        T_body2thrust   = orientation_transpose(np.ones_like(T_body2inertial[:])*body2thrust)  
        V_thrust        = orientation_product(T_body2thrust,V_body) 
        
        # Now just use the aligned velocity
        V     = V_thrust[:,0,None] 
        V_inf = V_thrust   
        ua    = np.zeros_like(V)
        ut    = np.zeros_like(V)
        
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
            lambda_i      = ua/(omega*R)
 
        #Things that don't change with iteration
        N       = len(c) # Number of stations     
        
        if  a_pol != None and a_loc != None:
            airfoil_polars = Data() 
            # check dimension of section
            if len(a_loc) != N:
                raise AssertionError('Dimension of airfoil sections must be equal to number of stations on rotor')
            # compute airfoil polars for airfoils 
            airfoil_polars = compute_airfoil_polars(self, a_geo, a_pol)
            airfoil_cl     = airfoil_polars.lift_coefficients
            airfoil_cd     = airfoil_polars.drag_coefficients
            AoA_sweep      = airfoil_polars.angle_of_attacks
        
        if self.radius_distribution is None:
            chi0    = Rh/R   # Where the rotor blade actually starts
            chi     = np.linspace(chi0,1,N+1)  # Vector of nondimensional radii
            chi     = chi[0:N]
        
        nu         = mu/rho                         
        lamda      = V/(omega*R)              # Speed ratio
        r          = chi*R                    # Radial coordinate
        pi         = np.pi
        pi2        = pi*pi
        x          = r*np.multiply(omega,1/V) # Nondimensional distance
        n          = omega/(2.*pi)            # Cycles per second
        J          = V/(2.*R*n)    
        blade_area = sp.integrate.cumtrapz(B*c, r-r[0])
        sigma      = blade_area[-1]/(pi*r[-1]**2)          
        omegar     = np.outer(omega,r)
        Ua         = np.outer((V + ua),np.ones_like(r))
        Ut         = omegar - ut
        U          = np.sqrt(Ua*Ua + Ut*Ut)
        
        #Things that will change with iteration
        size = (len(a),N)
        Cl   = np.zeros((1,N))  
        
        #Setup a Newton iteration
        psi    = np.ones(size)*0.5
        psiold = np.zeros(size)
        diff   = 1.
        
        ii    = 0
        broke = False 
        tol   = 1e-6    # Convergence tolerance         
        while (diff>tol):
            sin_psi = np.sin(psi)
            cos_psi = np.cos(psi)
            Wa      = 0.5*Ua + 0.5*U*sin_psi
            Wt      = 0.5*Ut + 0.5*U*cos_psi   
            va      = Wa - Ua
            vt      = Ut - Wt
            alpha   = beta - np.arctan2(Wa,Wt)
            W       = (Wa*Wa + Wt*Wt)**0.5
            Ma      = (W)/a #a is the speed of sound  
            lamdaw  = r*Wa/(R*Wt)
            
            # Limiter to keep from Nan-ing
            lamdaw[lamdaw<0.] = 0.
            
            f            = (B/2.)*(1.-r/R)/lamdaw
            piece        = np.exp(-f)
            arccos_piece = np.arccos(piece)
            F            = 2.*arccos_piece/pi
            Gamma        = vt*(4.*pi*r/B)*F*(1.+(4.*lamdaw*R/(pi*B*r))*(4.*lamdaw*R/(pi*B*r)))**0.5
            
            # Estimate Cl max
            Re         = (W*c)/nu 
            Cl_max_ref = -0.0009*tc**3 + 0.0217*tc**2 - 0.0442*tc + 0.7005
            Re_ref     = 9.*10**6      
            Cl1maxp    = Cl_max_ref * ( Re / Re_ref ) **0.1
            
            # Compute blade CL distribution from the airfoil data 
            if  a_pol != None and a_loc != None: 
                for k in range(N):
                    Cl[0,k] = np.interp(alpha[0,k],AoA_sweep,airfoil_cl[a_loc[k]])
            else:
                # If not airfoil polar provided, use 2*pi as lift curve slope
                Cl = 2.*pi*alpha
            
            # By 90 deg, it's totally stalled.
            Cl[Cl>Cl1maxp]  = Cl1maxp[Cl>Cl1maxp] # This line of code is what changed the regression testing
            Cl[alpha>=pi/2] = 0.
            
            # Scale for Mach, this is Karmen_Tsien
            Cl[Ma[:,:]<1.] = Cl[Ma[:,:]<1.]/((1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5+((Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])/(1+(1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5))*Cl[Ma<1.]/2)
            
            # If the blade segments are supersonic, don't scale
            Cl[Ma[:,:]>=1.] = Cl[Ma[:,:]>=1.] 
            
            Rsquiggly = Gamma - 0.5*W*c*Cl
            
            #An analytical derivative for dR_dpsi, this is derived by taking a derivative of the above equations
            #This was solved symbolically in Matlab and exported        
            f_wt_2 = 4*Wt*Wt
            f_wa_2 = 4*Wa*Wa
            Ucospsi  = U*cos_psi
            Usinpsi  = U*sin_psi
            Utcospsi = Ut*cos_psi
            Uasinpsi = Ua*sin_psi
            
            UapUsinpsi = (Ua + Usinpsi)
            utpUcospsi = (Ut + Ucospsi)
            
            utpUcospsi2 = utpUcospsi*utpUcospsi
            UapUsinpsi2 = UapUsinpsi*UapUsinpsi
            
            dR_dpsi = ((4.*U*r*arccos_piece*sin_psi*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5))/B - 
                       (pi*U*(Ua*cos_psi - Ut*sin_psi)*(beta - np.arctan((Wa+Wa)/(Wt+Wt))))/(2.*(f_wt_2 + f_wa_2)**(0.5))
                       + (pi*U*(f_wt_2 +f_wa_2)**(0.5)*(U + Utcospsi  +  Uasinpsi))/(2.*(f_wa_2/(f_wt_2) + 1.)*utpUcospsi2)
                       - (4.*U*piece*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5)*(R - r)*(Ut/2. - 
                      (Ucospsi)/2.)*(U + Utcospsi + Uasinpsi ))/(f_wa_2*(1. - np.exp(-(B*(Wt+Wt)*(R - 
                       r))/(r*(Wa+Wa))))**(0.5)) + (128.*U*r*arccos_piece*(Wa+Wa)*(Ut/2. - (Ucospsi)/2.)*(U + 
                       Utcospsi  + Uasinpsi ))/(BBB*pi2*utpUcospsi*utpUcospsi2*((16.*f_wa_2)/(BB*pi2*f_wt_2) + 1.)**(0.5))) 
            
            dR_dpsi[np.isnan(dR_dpsi)] = 0.1
                      
            dpsi   = -Rsquiggly/dR_dpsi
            psi    = psi + dpsi
            diff   = np.max(abs(psiold-psi))
            psiold = psi
            
            # If its really not going to converge
            if np.any(psi>(pi*85.0/180.)) and np.any(dpsi>0.0):
                broke = True
                break
                
            ii+=1
                
            if ii>2000:
                broke = True
                break
            
        # There is also RE scaling
        #This is an atrocious fit of DAE51 data at RE=50k for Cd
        Cdval = (0.108*(Cl*Cl*Cl*Cl)-0.2612*(Cl*Cl*Cl)+0.181*(Cl*Cl)-0.0139*Cl+0.0278)*((50000./Re)**0.2)
        Cdval[alpha>=pi/2] = 2.
        
        # More Cd scaling from Mach from AA241ab notes for turbulent skin friction
        Tw_Tinf = 1. + 1.78*(Ma*Ma)
        Tp_Tinf = 1. + 0.035*(Ma*Ma) + 0.45*(Tw_Tinf-1.)
        Tp      = (Tp_Tinf)*T
        Rp_Rinf = (Tp_Tinf**2.5)*(Tp+110.4)/(T+110.4)
        
        Cd = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval  
        epsilon  = Cd/Cl
        epsilon[epsilon==np.inf] = 10. 
        deltar   = (r[1]-r[0])
        thrust   = rho*B*(np.sum(Gamma*(Wt-epsilon*Wa)*deltar,axis=1)[:,None])
        torque   = rho*B*np.sum(Gamma*(Wa+epsilon*Wt)*r*deltar,axis=1)[:,None]
        D        = 2*R 
        tip_speed = omega*R
        
        # Leishman's thrust coefficient for rotor
        Ctl        = thrust/(rho*disk_area*( tip_speed*tip_speed))  # Eqn 2.36 Principles of Helicopter Aerodynamics 
        Ctl[Ctl<0] = 0.        # prevent things from breaking
        
        # motor thrust coefficient
        Ct       = thrust/(rho*(n*n)*(D*D*D*D)) # used for motor model
        Ct[Ct<0] = 0.  
        Cd0      = self.profile_drag_coefficient   
        Cp       = np.zeros_like(Ct)
        Cpl      = np.zeros_like(Ct)
        power    = np.zeros_like(Ct) 
        for i in range(len(Vv)): 
            if -1. < Vv[i][0] < 1.: # vertical/axial flight 
                Cpl[i]      = (kappa*(Ctl[i]**1.5)/(2**.5))+sigma*Cd0/8. # Eqn 2.43 Principles of Helicopter Aerodynamics 
                power[i]    = Cpl[i]*rho[i]*disk_area*(tip_speed[i]*tip_speed[i]*tip_speed[i]) 
                torque[i]   = power[i]/omega[i]    
            else:   
                power[i]    = torque[i]*omega[i]   
                torque[i]   = power[i]/omega[i]  
            Cp[i] = power[i]/(rho[i]*(n[i]*n[i]*n[i])*(D*D*D*D*D))  
        
        # torque coefficient 
        Cq = torque/(rho*(n*n)*(D*D*D*D)*R) 
        
        thrust[conditions.propulsion.throttle[:,0] <=0.0] = 0.0
        power[conditions.propulsion.throttle[:,0]  <=0.0] = 0.0 
        torque[conditions.propulsion.throttle[:,0]  <=0.0] = 0.0 
        thrust[omega<0.0] = - thrust[omega<0.0] 

        etap     = V*thrust/power     
        
        conditions.propulsion.etap = etap        
        
        # store data
        results_conditions                   = Data     
        outputs                              = results_conditions(
            num_blades                       = B,
            rotor_radius                     = R,
            rotor_diameter                   = D,
            number_sections                  = N,
            radius_distribution              = np.linspace(Rh ,R, N),
            chord_distribution               = c,     
            twist_distribution               = beta,            
            normalized_radial_distribution   = r,
            thrust_angle                     = theta,
            speed_of_sound                   = conditions.freestream.speed_of_sound,
            density                          = conditions.freestream.density,
            velocity                         = Vv, 
            tangential_velocity_distribution = vt, 
            axial_velocity_distribution      = va, 
            drag_coefficient                 = Cd,
            lift_coefficient                 = Cl,       
            omega                            = omega, 
            dT_dR                            = rho*(Gamma*(Wt-epsilon*Wa)),   
            dT_dr                            = rho*(Gamma*(Wt-epsilon*Wa))*R,  
            thrust_distribution              = rho*(Gamma*(Wt-epsilon*Wa))*deltar, 
            thrust_per_blade                 = thrust/B,  
            thrust_coefficient               = Ct,  
            dQ_dR                            = rho*(Gamma*(Wa+epsilon*Wt)*r), 
            dQ_dr                            = rho*(Gamma*(Wa+epsilon*Wt)*r)*R,
            torque_distribution              = rho*(Gamma*(Wa+epsilon*Wt)*r)*deltar,
            torque_per_blade                 = torque/B,   
            torque_coefficient               = Cq,   
            power                            = power,
            power_coefficient                = Cp, 
            mid_chord_aligment               = self.mid_chord_aligment     
        ) 
        
        return thrust, torque, power, Cp, outputs , etap
 