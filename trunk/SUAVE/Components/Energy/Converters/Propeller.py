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
        B      = self.number_blades
        R      = self.tip_radius
        Rh     = self.hub_radius
        beta   = self.twist_distribution
        c      = self.chord_distribution
        omega1 = self.inputs.omega 
        a_sec  = self.airfoil_sections        
        a_secl = self.airfoil_section_location        
        rho    = conditions.freestream.density[:,0,None]
        mu     = conditions.freestream.dynamic_viscosity[:,0,None]
        Vv     = conditions.frames.inertial.velocity_vector
        Vh     = self.induced_hover_velocity 
        a      = conditions.freestream.speed_of_sound[:,0,None]
        T      = conditions.freestream.temperature[:,0,None]
        theta  = self.thrust_angle
        tc     = .12 # Thickness to chord
        
        BB     = B*B
        BBB    = BB*B
            
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
            T_body2thrust      = orientation_transpose(body2thrust)
       
        V_thrust      = orientation_product(T_body2thrust,V_body)
        
        # Now just use the aligned velocity
        V = V_thrust[:,0,None]
        
        ua = np.zeros_like(V)
        if Vh != None:   
            for i in range(len(V)):
                V_inf = V_thrust[i] 
                V_Vh =  V_thrust[i][0]/Vh
                if Vv[i,:].all()  == True :
                    ua[i] = Vh
                elif Vv[i][0]  == 0 and  Vv[i][2] != 0: # vertical / axial flight
                    if V_Vh > 0: # climbing 
                        ua[i] = Vh*(-(-V_inf[0]/(2*Vh)) + np.sqrt((-V_inf[0]/(2*Vh))**2 + 1))
                    elif -2 <= V_Vh and V_Vh <= 0:  # slow descent                 
                        ua[i] = Vh*(1.15 -1.125*(V_Vh) - 1.372*(V_Vh)**2 - 1.718*(V_Vh)**2 - 0.655*(V_Vh)**4 ) 
                    else: # windmilling 
                        print("rotor is in the windmill break state!")
                        ua[i] = Vh*(-(-V_inf[0]/(2*Vh)) - np.sqrt((-V_inf[0]/(2*Vh))**2 + 1))
                else: # forward flight conditions                 
                    func = lambda vi: vi - (Vh**2)/(np.sqrt(((-V_inf[2])**2 + (V_inf[0] + vi)**2)))
                    vi_initial_guess = V_inf[0]
                    ua[i]    = fsolve(func,vi_initial_guess)
        else: 
            ua = 0.0 
 
        ut = 0.0
        
        nu    = mu/rho
        tol   = 1e-5 # Convergence tolerance
        
        omega = omega1*1.0
        omega = np.abs(omega)
        
        #Things that don't change with iteration
        N       = len(c) # Number of stations     
        
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
        
        if self.radius_distribution is None:
            chi0    = Rh/R   # Where the propeller blade actually starts
            chi     = np.linspace(chi0,1,N+1)  # Vector of nondimensional radii
            chi     = chi[0:N]
        
        else:
            chi = self.radius_distribution
        
        lamda   = V/(omega*R)              # Speed ratio
        r       = chi*R                    # Radial coordinate
        pi      = np.pi
        pi2     = pi*pi
        x       = r*np.multiply(omega,1/V) # Nondimensional distance
        n       = omega/(2.*pi)            # Cycles per second
        J       = V/(2.*R*n)    
        #sigma   = np.multiply(B*c,1./(2.*pi*r))
        blade_area = sp.integrate.cumtrapz(B*c, r-r[0])
        sigma   = blade_area[-1]/(pi*r[-1]**2)   
        
        omegar = np.outer(omega,r)
        Ua = np.outer((V + ua),np.ones_like(r))
        Ut = omegar - ut
        U  = np.sqrt(Ua*Ua + Ut*Ut)
        
        #Things that will change with iteration
        size = (len(a),N)
    
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
            va     = Wa - Ua
            vt      = Ut - Wt
            alpha   = beta - np.arctan2(Wa,Wt)
            W       = (Wa*Wa + Wt*Wt)**0.5
            Ma      = (W)/a #a is the speed of sound
            
            #if np.any(Ma> 1.0):
                #warn('Propeller blade tips are supersonic.', Warning)
            
            lamdaw = r*Wa/(R*Wt)
            
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
            
            # Ok, from the airfoil data, given Re, Ma, alpha we need to find Cl
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
        Ct       = thrust/(rho*(n*n)*(D*D*D*D))
        Ct[Ct<0] = 0.        #prevent things from breaking
        kappa    = self.induced_power_factor 
        Cd0      = self.profile_drag_coefficient   
        Cp    = np.zeros_like(Ct)
        power = np.zeros_like(Ct)        
        for i in range(len(Vv)):
            if -1. <Vv[i][0] <1.: # vertical/axial flight
                Cp[i]       = (kappa*(Ct[i]**1.5)/(2**.5))+sigma*Cd0/8.
                power[i]    = Cp[i]*(rho[i]*(n[i]*n[i]*n[i])*(D*D*D*D*D))
                torque[i]   = power[i]/omega[i]  
            else:  
                power[i]    = torque[i]*omega[i]   
                Cp[i]       = power[i]/(rho[i]*(n[i]*n[i]*n[i])*(D*D*D*D*D))


        
  
        thrust[conditions.propulsion.throttle[:,0] <=0.0] = 0.0
        power[conditions.propulsion.throttle[:,0]  <=0.0] = 0.0
        
        thrust[omega1<0.0] = - thrust[omega1<0.0]

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
            twist_distribution        = beta,            
            r0                        = r,
            thrust_angle              = theta,
            speed_of_sound            = conditions.freestream.speed_of_sound,
            density                   = conditions.freestream.density,
            velocity                  = Vv, 
            vt                        = vt, 
            va                        = va, 
            drag_coefficient          = Cd,
            lift_coefficient          = Cl,       
            omega                     = omega,          
            
            blade_dT_dR               = rho*(Gamma*(Wt-epsilon*Wa)),   
            blade_dT_dr               = rho*(Gamma*(Wt-epsilon*Wa))*R,  
            blade_T_distribution      = rho*(Gamma*(Wt-epsilon*Wa))*deltar, 
            blade_T                   = thrust/B,  
            Ct                        = 0, 
            Cts                       = 0, 
            
            blade_dQ_dR               = rho*(Gamma*(Wa+epsilon*Wt)*r), 
            blade_dQ_dr               = rho*(Gamma*(Wa+epsilon*Wt)*r)*R,
            blade_Q_distribution      = rho*(Gamma*(Wa+epsilon*Wt)*r)*deltar,
            blade_Q                   = torque/B,   
            Cq                        = 0, 
            
            power                     = power,
            
            mid_chord_aligment        = self.mid_chord_aligment     
        ) 
        
        return thrust, torque, power, Cp, outputs  , etap  


    

    def spin_variable_pitch(self,conditions):
        """ Analyzes a propeller given geometry and operating conditions
                 
                 Inputs:
                     hub radius
                     tip radius
                     rotation rate
                     freestream velocity
                     number of blades
                     number of stations
                     chord distribution
                     twist distribution
                     airfoil data
       
                 Outputs:
                     Power coefficient
                     Thrust coefficient
                     
                 Assumptions:
                     Based on Qprop Theory document
       
           """
           
        #Unpack            
        B      = self.number_blades
        R      = self.tip_radius
        Rh     = self.hub_radius
        beta_in = self.twist_distribution
        c      = self.chord_distribution
        omega1 = self.inputs.omega 
        a_sec  = self.airfoil_sections        
        a_secl = self.airfoil_section_location        
        rho    = conditions.freestream.density[:,0,None]
        mu     = conditions.freestream.dynamic_viscosity[:,0,None]
        Vv     = conditions.frames.inertial.velocity_vector
        Vh     = self.induced_hover_velocity 
        a      = conditions.freestream.speed_of_sound[:,0,None]
        T      = conditions.freestream.temperature[:,0,None]
        theta  = self.thrust_angle
        tc     = .12 # Thickness to chord
        beta_c  = conditions.propulsion.pitch_command
        ducted  = self.ducted
        
        beta   = beta_in + beta_c        
        
        BB     = B*B
        BBB    = BB*B
            
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
            T_body2thrust      = orientation_transpose(body2thrust)
       
        V_thrust      = orientation_product(T_body2thrust,V_body)
        
        # Now just use the aligned velocity
        V = V_thrust[:,0,None]
        
        ua = np.zeros_like(V)
        if Vh != None:   
            for i in range(len(V)):
                V_inf = V_thrust[i] 
                V_Vh =  V_thrust[i][0]/Vh
                if Vv[i,:].all()  == True :
                    ua[i] = Vh
                elif Vv[i][0]  == 0 and  Vv[i][2] != 0: # vertical / axial flight
                    if V_Vh > 0: # climbing 
                        ua[i] = Vh*(-(-V_inf[0]/(2*Vh)) + np.sqrt((-V_inf[0]/(2*Vh))**2 + 1))
                    elif -2 <= V_Vh and V_Vh <= 0:  # slow descent                 
                        ua[i] = Vh*(1.15 -1.125*(V_Vh) - 1.372*(V_Vh)**2 - 1.718*(V_Vh)**2 - 0.655*(V_Vh)**4 ) 
                    else: # windmilling 
                        print("rotor is in the windmill break state!")
                        ua[i] = Vh*(-(-V_inf[0]/(2*Vh)) - np.sqrt((-V_inf[0]/(2*Vh))**2 + 1))
                else: # forward flight conditions                 
                    func = lambda vi: vi - (Vh**2)/(np.sqrt(((-V_inf[2])**2 + (V_inf[0] + vi)**2)))
                    vi_initial_guess = V_inf[0]
                    ua[i]    = fsolve(func,vi_initial_guess)
        else: 
            ua = 0.0 
 
        ut = 0.0
        
        nu    = mu/rho
        tol   = 1e-5 # Convergence tolerance
        
        omega = omega1*1.0
        omega = np.abs(omega)
        
        #Things that don't change with iteration
        N       = len(c) # Number of stations     
        
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
        
        if self.radius_distribution is None:
            chi0    = Rh/R   # Where the propeller blade actually starts
            chi     = np.linspace(chi0,1,N+1)  # Vector of nondimensional radii
            chi     = chi[0:N]
        
        else:
            chi = self.radius_distribution
        
        lamda   = V/(omega*R)              # Speed ratio
        r       = chi*R                    # Radial coordinate
        pi      = np.pi
        pi2     = pi*pi
        x       = r*np.multiply(omega,1/V) # Nondimensional distance
        n       = omega/(2.*pi)            # Cycles per second
        J       = V/(2.*R*n)    
        #sigma   = np.multiply(B*c,1./(2.*pi*r))
        blade_area = sp.integrate.cumtrapz(B*c, r-r[0])
        sigma   = blade_area[-1]/(pi*r[-1]**2)   
        
        omegar = np.outer(omega,r)
        Ua = np.outer((V + ua),np.ones_like(r))
        Ut = omegar - ut
        U  = np.sqrt(Ua*Ua + Ut*Ut)
        
        #Things that will change with iteration
        size = (len(a),N)
    
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
            va     = Wa - Ua
            vt      = Ut - Wt
            alpha   = beta - np.arctan2(Wa,Wt)
            W       = (Wa*Wa + Wt*Wt)**0.5
            Ma      = (W)/a #a is the speed of sound
            
            #if np.any(Ma> 1.0):
                #warn('Propeller blade tips are supersonic.', Warning)
            
            lamdaw = r*Wa/(R*Wt)
            
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
            
            # Ok, from the airfoil data, given Re, Ma, alpha we need to find Cl
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
        Ct       = thrust/(rho*(n*n)*(D*D*D*D))
        Ct[Ct<0] = 0.        #prevent things from breaking
        kappa    = self.induced_power_factor 
        Cd0      = self.profile_drag_coefficient   
        Cp    = np.zeros_like(Ct)
        power = np.zeros_like(Ct)        
        for i in range(len(Vv)):
            if -1. <Vv[i][0] <1.: # vertical/axial flight
                Cp[i]       = (kappa*(Ct[i]**1.5)/(2**.5))+sigma*Cd0/8.
                power[i]    = Cp[i]*(rho[i]*(n[i]*n[i]*n[i])*(D*D*D*D*D))
                torque[i]   = power[i]/omega[i]  
            else:  
                power[i]    = torque[i]*omega[i]   
                Cp[i]       = power[i]/(rho[i]*(n[i]*n[i]*n[i])*(D*D*D*D*D))


        
  
        thrust[conditions.propulsion.throttle[:,0] <=0.0] = 0.0
        power[conditions.propulsion.throttle[:,0]  <=0.0] = 0.0
        
        thrust[omega1<0.0] = - thrust[omega1<0.0]

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
            twist_distribution        = beta,            
            r0                        = r,
            thrust_angle              = theta,
            speed_of_sound            = conditions.freestream.speed_of_sound,
            density                   = conditions.freestream.density,
            velocity                  = Vv, 
            vt                        = vt, 
            va                        = va, 
            drag_coefficient          = Cd,
            lift_coefficient          = Cl,       
            omega                     = omega,          
            
            blade_dT_dR               = rho*(Gamma*(Wt-epsilon*Wa)),   
            blade_dT_dr               = rho*(Gamma*(Wt-epsilon*Wa))*R,  
            blade_T_distribution      = rho*(Gamma*(Wt-epsilon*Wa))*deltar, 
            blade_T                   = thrust/B,  
            Ct                        = 0, 
            Cts                       = 0, 
            
            blade_dQ_dR               = rho*(Gamma*(Wa+epsilon*Wt)*r), 
            blade_dQ_dr               = rho*(Gamma*(Wa+epsilon*Wt)*r)*R,
            blade_Q_distribution      = rho*(Gamma*(Wa+epsilon*Wt)*r)*deltar,
            blade_Q                   = torque/B,   
            Cq                        = 0, 
            
            power                     = power,
            
            mid_chord_aligment        = self.mid_chord_aligment     
        ) 
        
        return thrust, torque, power, Cp, outputs  , etap  

    
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
