## @ingroup Components-Energy-Converters
# Rotor.py
#
# Created:  Jun 2014, E. Botero
# Modified: Jan 2016, T. MacDonald
#           Feb 2019, M. Vegh            
#           Mar 2020, M. Clarke
#           Aug 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Core import Data
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_polars \
     import compute_airfoil_polars
from SUAVE.Methods.Geometry.Three_Dimensional \
     import  orientation_product, orientation_transpose

# package imports
import numpy as np
import scipy as sp
from scipy.optimize import fsolve


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
        self.use_Blade_Element_Theory = False 
        self.number_azimuthal_stations= 24
        self.induced_power_factor     = 1.48  #accounts for interference effects
        self.profile_drag_coefficient = .03        
        self.tag                      = 'Rotor'


    def spin(self,conditions):
        """Analyzes a rotor given geometry and operating conditions.

        Assumptions:
        per source

        Source:
        Drela, M. "Qprop Formulation", MIT AeroAstro, June 2006
        http://web.mit.edu/drela/Public/web/qprop/qprop_theory.pdf
        
        Leishman, Gordon J. Principles of helicopter aerodynamics
        Cambridge university press, 2006.      

        Inputs:
        self.inputs.omega                    [radian/s]
        conditions.freestream.               
          density                            [kg/m^3]
          dynamic_viscosity                  [kg/(m-s)]
          speed_of_sound                     [m/s]
          temperature                        [K]
        conditions.frames.                   
          body.transform_to_inertial         (rotation matrix)
          inertial.velocity_vector           [m/s]
        conditions.propulsion.               
          throttle                           [-]
                                             
        Outputs:                             
        conditions.propulsion.outputs.        
           number_radial_stations            [-]
           number_azimuthal_stations         [-] 
           disc_radial_distribution          [m]
           thrust_angle                      [rad]
           speed_of_sound                    [m/s]
           density                           [kg/m-3]
           velocity                          [m/s]
           disc_tangential_induced_velocity  [m/s]
           disc_axial_induced_velocity       [m/s]
           disc_tangential_velocity          [m/s]
           disc_axial_velocity               [m/s]
           drag_coefficient                  [-]
           lift_coefficient                  [-]
           omega                             [rad/s]
           disc_circulation                  [-] 
           blade_dT_dR                       [N/m]
           blade_dT_dr                       [N]
           blade_thrust_distribution         [N]
           disc_thrust_distribution          [N]
           thrust_per_blade                  [N]
           thrust_coefficient                [-] 
           azimuthal_distribution            [rad]
           disc_azimuthal_distribution       [rad]
           blade_dQ_dR                       [N]
           blade_dQ_dr                       [Nm]
           blade_torque_distribution         [Nm] 
           disc_torque_distribution          [Nm] 
           torque_per_blade                  [Nm] 
           torque_coefficient                [-] 
           power                             [W]    
           power_coefficient                 [-] 
                                             
        Properties Used:                     
        self.                                
          number_blades                      [-]
          tip_radius                         [m]
          hub_radius                         [m]
          twist_distribution                 [radians]
          chord_distribution                 [m]
          mid_chord_aligment                 [m] 
          thrust_angle                       [radians]
        """         
           
        #Unpack    
        B       = self.number_blades
        R       = self.tip_radius
        Rh      = self.hub_radius
        beta_0  = self.twist_distribution
        c       = self.chord_distribution
        chi     = self.radius_distribution
        omega   = self.inputs.omega 
        a_geo   = self.airfoil_geometry      
        a_loc   = self.airfoil_polar_stations  
        cl_sur  = self.airfoil_cl_surrogates
        cd_sur  = self.airfoil_cd_surrogates   
        
        rho     = conditions.freestream.density[:,0,None]
        mu      = conditions.freestream.dynamic_viscosity[:,0,None]
        Vv      = conditions.frames.inertial.velocity_vector
        Vh      = self.induced_hover_velocity 
        a       = conditions.freestream.speed_of_sound[:,0,None]
        T       = conditions.freestream.temperature[:,0,None]
        theta   = self.thrust_angle
        tc      = self.thickness_to_chord  
        sigma   = self.blade_solidity   
        Na      = self.number_azimuthal_stations
        use_BET = self.use_Blade_Element_Theory
        BB      = B*B    
        BBB     = BB*B
    
        try:
            pitch_command     = conditions.propulsion.pitch_command
            total_blade_pitch = beta_0 + pitch_command   
        except:
            total_blade_pitch = beta_0 
        
        try:
            theta = self.propeller_thrust_angle
        except:
            pass
        
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
        V        = V_thrust[:,0,None] 
        V_inf    = V_thrust 
        ua       = np.zeros_like(V)              
        ut       = np.zeros_like(V) 
    
        #Things that don't change with iteration
        Nr       = len(c) # Number of stations radially    
        ctrl_pts = len(Vv)
                 
        # set up non dimensional radial distribution 
        if self.radius_distribution is None:
            chi0= Rh/R                      # Where the rotor blade actually starts
            chi = np.linspace(chi0,1,Nr+1)  # Vector of nondimensional radii
            chi = chi[0:Nr]
    
        else:
            chi = self.radius_distribution/R
    
        omega          = np.abs(omega)        
        r              = chi*R                              # Radial coordinate 
        pi             = np.pi                              
        A              = pi*(R**2)                          
        n              = omega/(2.*pi)                      # Cycles per second  
        nu             = mu/rho     
        blade_area     = sp.integrate.cumtrapz(B*c, r-r[0]) # blade area 
        sigma          = blade_area[-1]/(pi*r[-1]**2)       # solidity   # (page 28 Leishman)      
    
        # blade flap rate and sweep(cone) angle 
        beta_blade_dot = 0  # currently no flaping 
        beta_blade     = 0  # currently no coning            
    
        # azimuth distribution 
        psi            = np.linspace(0,2*pi,Na+1)[:-1]
        psi_2d         = np.tile(np.atleast_2d(psi).T,(1,Nr))
        psi_2d         = np.repeat(psi_2d[np.newaxis, :, :], ctrl_pts, axis=0)  
    
        # 2 dimensiona radial distribution non dimensionalized
        chi_2d         = np.tile(chi ,(Na,1))            
        chi_2d         = np.repeat(chi_2d[ np.newaxis,:, :], ctrl_pts, axis=0) 
                       
        r_dim_2d       = np.tile(r ,(Na,1))  
        r_dim_2d       = np.repeat(r_dim_2d[ np.newaxis,:, :], ctrl_pts, axis=0)  
    
        # Momentum theory approximation of inflow for BET if the advance ratio is large
        edgewise = abs(V_thrust[:,0]/V_thrust[:,2])
        if any(edgewise[:] < 10.0) or use_BET:
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
                lambda_i  = ua/(omega*R)            
            
            # compute lambda and mu 
            lambda_tot     = (np.atleast_2d(V_inf[:,0]).T + ua)/(omega*R)       # inflow advance ratio (page 30 Leishman)
            mu_prop        = (np.atleast_2d(V_inf[:,2]).T) /(omega*R)           # rotor advance ratio  (page 30 Leishman)
            lambda_c       = (np.atleast_2d(V_inf[:,0]).T)/(omega*R)            # normal velocity ratio (page 30 Leishman)
            lambda_i       = ua/(omega*R)                                       # induced inflow ratio  (page 30 Leishman)
        
            # wake skew angle 
            X              = np.arctan(mu_prop/lambda_tot)     
            
            # 2-D chord distribution 
            chord       = np.tile(c,(Na,1))  
            chord_2d    = np.repeat(chord[np.newaxis,:, :], ctrl_pts, axis=0)
    
            # 2-D blade twist distribution 
            theta_2d    = np.tile(total_blade_pitch,(Na ,1))
            theta_2d    = np.repeat(theta_2d[np.newaxis,:, :], ctrl_pts, axis=0)    
    
            # 2-D inflow ratio 
            mu_2d       = np.tile(np.atleast_2d(mu_prop),(1,Nr))
            mu_2d       = np.repeat(mu_2d[:, np.newaxis,  :], Na, axis=1)         
    
            # 2-D inflow ratio 
            lambda_2d   = np.tile(np.atleast_2d(lambda_tot),(1,Nr))
            lambda_2d   = np.repeat(lambda_2d[:, np.newaxis,  :], Na, axis=1)       
    
            # wake skew angle 
            X           = np.arctan(mu_2d/lambda_2d) # wake skew angle (Section 5.2 page 133 Leishman) 
            kx_2d       = np.tan(X/2)             # slope of downwash at center of rotor disk for forward flight eqn 5.42 (page 136 Leishman)        
            #kx_2d       = (4/3)*((1.8*mu_2d**2)*np.sqrt(1 + (lambda_2d/mu_2d)**2)  - lambda_2d/mu_2d)  # eqn 5.43 (page 136 Leishman)
            ky_2d       = -2*mu_2d                # eqn 5.44 (page 136 Leishman) 
    
            lambda_i_2d = np.tile(np.atleast_2d(lambda_i),(1,Nr))
            lambda_i_2d = np.repeat(lambda_i_2d[:, np.newaxis,  :], Na, axis=1) 
    
            lambda_c_2d = np.tile(np.atleast_2d(lambda_c),(1,Nr))
            lambda_c_2d = np.repeat(lambda_c_2d[:, np.newaxis,  :], Na, axis=1)         
    
            # motification to initial radial inflow distribution  
            lambda_i_2d = lambda_i_2d*(1 + kx_2d*chi_2d*np.cos(psi_2d) + ky_2d*chi_2d*np.sin(psi_2d) )  # eqn 5.41 (page 136 Leishman)  
            lambda_2d   = lambda_c_2d + lambda_i_2d
    
            omega_2d    = np.tile(np.atleast_2d(omega),(1,Nr))
            omega_2d    = np.repeat(omega[:, np.newaxis,  :], Na, axis=1)   
            Va_ind_2d   = lambda_i_2d*(omega_2d*R)     
            Vt_ind_2d   = np.zeros_like(Va_ind_2d)            
            
            # axial, tangential and radial components of local blade flow [multiplied by omega*R to dimensionalize] 
            omega_R_2d  = np.tile(np.atleast_2d(omega*R),(1,Nr))
            omega_R_2d  = np.repeat(omega_R_2d[:, np.newaxis,  :], Na, axis=1)  
            Vt_2d       = omega_R_2d * (chi_2d  + mu_2d*np.sin(psi_2d))                                        # velocity tangential to the disk plane, positive toward the trailing edge eqn 6.34 pg 165           
            Vr_2d       = omega_R_2d * (mu_2d*np.cos(psi_2d))                                                  # radial velocity , positive outward   eqn 6.35 pg 165                 
            Va_2d       = omega_R_2d * (lambda_2d + chi_2d *beta_blade_dot + beta_blade*mu_2d*np.cos(psi_2d))  # velocity perpendicular to the disk plane, positive downward  eqn 6.36 pg 166  
    
            # local total velocity 
            U_2d        = np.sqrt(Vt_2d**2 + Va_2d**2) # (page 165 Leishman)
    
            # blade incident angle 
            phi_2d      = np.arctan(Va_2d/Vt_2d)       # (page 166 Leishman)
    
            # local blade angle of attack
            alpha       = theta_2d - phi_2d            # (page 166 Leishman)
    
            # Scale for Mach, this is Karmen_Tsien 
            a_2d        = np.tile(np.atleast_2d(a),(1,Nr))
            a_2d        = np.repeat(a_2d[:, np.newaxis,  :], Na, axis=1)  
            
            # local mach number
            Ma          = (U_2d)/a_2d   
            
            # Estimate Cl max  
            nu_2d       = np.tile(np.atleast_2d(nu),(1,Nr))
            nu_2d       = np.repeat(nu_2d[:, np.newaxis,  :], Na, axis=1)   
            Re          = (U_2d*chord_2d)/nu_2d  
            
            # Compute blade Cl and Cd distribution from the airfoil data if provided else use thin airfoil theory  
            Cl          = np.zeros((ctrl_pts,Na,Nr))              
            Cdval       = np.zeros((ctrl_pts,Na,Nr))                 
            for ii in range(ctrl_pts):
                for jj in range(Nr):                 
                    Cl[ii,:,jj]    = cl_sur[a_geo[a_loc[jj]]](Re[ii,:,jj],alpha[ii,:,jj],grid=False)  
                    Cdval[ii,:,jj] = cd_sur[a_geo[a_loc[jj]]](Re[ii,:,jj],alpha[ii,:,jj],grid=False)  
    
            #More Cd scaling from Mach from AA241ab notes for turbulent skin friction 
            T_2d         = np.tile(np.atleast_2d(T),(1,Nr))
            T_2d         = np.repeat(T_2d[:, np.newaxis,  :], Na, axis=1)     
            Tw_Tinf      = 1. + 1.78*(Ma*Ma)
            Tp_Tinf      = 1. + 0.035*(Ma*Ma) + 0.45*(Tw_Tinf-1.)
            Tp           = (Tp_Tinf)*T_2d
            Rp_Rinf      = (Tp_Tinf**2.5)*(Tp+110.4)/(T_2d+110.4) 
            Cd           = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval  
    
            # local blade lift and drag  
            rho_2d       = np.tile(np.atleast_2d(rho),(1,Nr))
            rho_2d       = np.repeat(rho_2d[:, np.newaxis,  :], Na, axis=1)    
            dL           = 0.5 * rho_2d * U_2d**2 * chord_2d * Cl # eqn 6.37 (page 167 Leishman) 
            dD           = 0.5 * rho_2d * U_2d**2 * chord_2d * Cd # eqn 6.38 (page 167 Leishman) 
    
            # application of tip loss factor 
            tip_loss_factor            = 0.97 # (page 67 and  Leishman) make a property of the rotor
            dL[chi_2d>tip_loss_factor] = 0  # (page 63 & 90 and  Leishman) 
    
            # normal and tangential forces  
            dFz  = dL*np.cos(phi_2d) - dD*np.sin(phi_2d) # eqn 6.39 (page 167 Leishman) 
            dFx  = dL*np.sin(phi_2d) - dD*np.cos(phi_2d) # eqn 6.40 (page 167 Leishman)
    
            # average thrust and torque over aximuth
            deltar                  = chi_2d[:,:,1]-chi_2d[:,:,0]  
            deltar_2d               = np.repeat(deltar[:,  :, np.newaxis], Nr, axis=2) 
            blade_T_distribution    = np.mean((dFz*deltar_2d), axis = 1)
            blade_Q_distribution    = np.mean((dFx*chi_2d*deltar_2d), axis = 1)
            thrust                  = np.atleast_2d((B * np.sum(blade_T_distribution, axis = 1))).T 
            torque                  = np.atleast_2d((B * np.sum(blade_Q_distribution, axis = 1))).T 
            blade_T_distribution_2d = dFz*deltar_2d
            blade_Q_distribution_2d = dFx*chi_2d*deltar_2d  
            
            blade_dT_dR = np.zeros((len(Vv),Nr))
            blade_dT_dr = np.zeros_like(blade_dT_dR)
            blade_dQ_dR = np.zeros_like(blade_dT_dR)
            blade_dQ_dr = np.zeros_like(blade_dT_dR)
             
            blade_Gamma_2d  = 0.5*U_2d*chord_2d*Cl       
            
            Vt_ind_avg = np.mean(Vt_ind_2d , axis = 1)
            Va_ind_avg = np.mean(Va_ind_2d , axis = 1)
            Vt_avg     = np.mean(Vt_2d  , axis = 1)
            Va_avg     = np.mean(Va_2d , axis = 1)
            
            for i in range(len(Vv)):
                blade_dT_dR[i,:] = np.gradient(blade_T_distribution[i], deltar[i,0]*R) 
                blade_dT_dr[i,:] = np.gradient(blade_T_distribution[i], deltar[i,0])
                blade_dQ_dR[i,:] = np.gradient(blade_Q_distribution[i], deltar[i,0]*R)
                blade_dQ_dr[i,:] = np.gradient(blade_Q_distribution[i], deltar[i,0])   
    
        # Blade Element Momentum Theory 
        else:   
            
            # Things that will change with iteration
            size   = (len(a),Nr)
            omegar = np.outer(omega,r)
            Ua     = np.outer((V + ua),np.ones_like(r))
            Ut     = omegar - ut
            U      = np.sqrt(Ua*Ua + Ut*Ut)
            pi2    = pi*pi
            beta   = total_blade_pitch
            
            # Setup a Newton iteration
            PSI    = np.ones(size)
            PSIold = np.zeros(size)
            diff   = 1. 
            ii     = 0
            broke  = False
            tol    = 1e-6  # Convergence tolerance
            
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
                F            = 2.*arccos_piece/pi
                Gamma        = vt*(4.*pi*r/B)*F*(1.+(4.*lamdaw*R/(pi*B*r))*(4.*lamdaw*R/(pi*B*r)))**0.5 
                Re           = (W*c)/nu  
        
                Cl    = np.zeros((ctrl_pts,Nr))              
                Cdval = np.zeros((ctrl_pts,Nr)) 
                for jj in range(Nr):                 
                    Cl[:,jj]    = cl_sur[a_geo[a_loc[jj]]](Re[:,jj],alpha[:,jj],grid=False)  
                    Cdval[:,jj] = cd_sur[a_geo[a_loc[jj]]](Re[:,jj],alpha[:,jj],grid=False)    
                    
                # More Cd scaling from Mach from AA241ab notes for turbulent skin friction
                Tw_Tinf     = 1. + 1.78*(Ma*Ma)
                Tp_Tinf     = 1. + 0.035*(Ma*Ma) + 0.45*(Tw_Tinf-1.)
                Tp          = (Tp_Tinf)*T
                Rp_Rinf     = (Tp_Tinf**2.5)*(Tp+110.4)/(T+110.4) 
                Cd          = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval  
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
        
                # If its really not going to converge
                if np.any(PSI>pi/2) and np.any(dpsi>0.0):
                    print("Rotor BEMT did not converge to a solution")
                    break
        
                ii+=1 
                if ii>10000:
                    broke = True
                    print("Rotor BEMT did not converge to a solution")
                    break
    
            epsilon                  = Cd/Cl
            epsilon[epsilon==np.inf] = 10. 
            deltar                   = (r[1]-r[0])  
            
            blade_T_distribution     = rho*(Gamma*(Wt-epsilon*Wa))*deltar 
            blade_Q_distribution     = rho*(Gamma*(Wa+epsilon*Wt)*r)*deltar 
            thrust                   = rho*B*(np.sum(Gamma*(Wt-epsilon*Wa)*deltar,axis=1)[:,None])
            torque                   = rho*B*np.sum(Gamma*(Wa+epsilon*Wt)*r*deltar,axis=1)[:,None] 
            Va_2d                    = np.repeat(Wa.T[ : , np.newaxis , :], Na, axis=1).T
            Vt_2d                    = np.repeat(Wt.T[ : , np.newaxis , :], Na, axis=1).T
            Vt_ind_2d                = np.repeat(va.T[ : , np.newaxis , :], Na, axis=1).T
            Va_ind_2d                = np.repeat(vt.T[ : , np.newaxis , :], Na, axis=1).T
            blade_T_distribution_2d  = np.repeat(blade_T_distribution.T[ np.newaxis,:  , :], Na, axis=0).T 
            blade_Q_distribution_2d  = np.repeat(blade_Q_distribution.T[ np.newaxis,:  , :], Na, axis=0).T 
            
            blade_Gamma_2d           = np.repeat(Gamma.T[ : , np.newaxis , :], Na, axis=1).T
            blade_dT_dR              = rho*(Gamma*(Wt-epsilon*Wa))
            blade_dT_dr              = rho*(Gamma*(Wt-epsilon*Wa))*R
            blade_dQ_dR              = rho*(Gamma*(Wa+epsilon*Wt)*r)
            blade_dQ_dr              = rho*(Gamma*(Wa+epsilon*Wt)*r)*R
            
            Vt_ind_avg = vt
            Va_ind_avg = va
            Vt_avg     = Wt
            Va_avg     = Wa
            
        psi_2d   = np.repeat(np.atleast_2d(psi).T[np.newaxis,: ,:], Na, axis=0).T        
        D        = 2*R 
        Cq       = torque/(rho*A*R*(omega*R)**2)
        Ct       = thrust/(rho*A*(omega*R)**2)
        Ct[Ct<0] = 0.     # prevent things from breaking
        kappa    = self.induced_power_factor 
        Cd0      = self.profile_drag_coefficient   
        Cp       = np.zeros_like(Ct)
        power    = np.zeros_like(Ct)      
    
        for i in range(len(Vv)):   
            if -1. <Vv[i][0] <1.: # vertical/axial flight
                Cp[i]     = (kappa*(Ct[i]**1.5)/(2**.5))+sigma*Cd0/8.
                power[i]  = Cp[i]*(rho[i]*(n[i]*n[i]*n[i])*(D*D*D*D*D))
                torque[i] = power[i]/omega[i]  
            else:         
                power[i]  = torque[i]*omega[i]
                Cp[i]     = power[i]/(rho[i]*(n[i]*n[i]*n[i])*(D*D*D*D*D))
 
        Cq = torque/(rho*(n*n)*(D*D*D*D)*R) # torque coefficient 

        thrust[conditions.propulsion.throttle[:,0] <=0.0]  = 0.0
        power[conditions.propulsion.throttle[:,0]  <=0.0]  = 0.0 
        torque[conditions.propulsion.throttle[:,0]  <=0.0] = 0.0
        thrust[omega<0.0] = - thrust[omega<0.0]

        etap                                  = V*thrust/power 
        conditions.propulsion.etap            = etap   
        
        # store data
        self.azimuthal_distribution                   = psi  
        results_conditions                            = Data     
        outputs                                       = results_conditions( 
                    number_radial_stations            = Nr,
                    number_azimuthal_stations         = Na,   
                    disc_radial_distribution          = r_dim_2d,  
                    thrust_angle                      = theta,
                    speed_of_sound                    = conditions.freestream.speed_of_sound,
                    density                           = conditions.freestream.density,
                    velocity                          = Vv, 
                    blade_tangential_induced_velocity = Vt_ind_avg, 
                    blade_axial_induced_velocity      = Va_ind_avg,  
                    blade_tangential_velocity         = Vt_avg, 
                    blade_axial_velocity              = Va_avg,  
                    disc_tangential_induced_velocity  = Vt_ind_2d, 
                    disc_axial_induced_velocity       = Va_ind_2d,  
                    disc_tangential_velocity          = Vt_2d, 
                    disc_axial_velocity               = Va_2d, 
                    drag_coefficient                  = Cd,
                    lift_coefficient                  = Cl,       
                    omega                             = omega,  
                    disc_circulation                  = blade_Gamma_2d,
                    blade_dT_dR                       = blade_dT_dR,
                    blade_dT_dr                       = blade_dT_dr,
                    blade_thrust_distribution         = blade_T_distribution, 
                    disc_thrust_distribution          = blade_T_distribution_2d, 
                    thrust_per_blade                  = thrust/B, 
                    thrust_coefficient                = Ct, 
                    disc_azimuthal_distribution       = psi_2d ,
                    blade_dQ_dR                       = blade_dQ_dR,
                    blade_dQ_dr                       = blade_dQ_dr,
                    blade_torque_distribution         = blade_Q_distribution, 
                    disc_torque_distribution          = blade_Q_distribution_2d, 
                    torque_per_blade                  = torque/B,   
                    torque_coefficient                = Cq,   
                    power                             = power,
                    power_coefficient                 = Cp,                      
            ) 
    
        return thrust, torque, power, Cp, outputs , etap
