## @ingroup Components-Energy-Converters
# Propeller.py
#
# Created:  Jun 2014, E. Botero
# Modified: Jan 2016, T. MacDonald
#           Feb 2019, M. Vegh            
#           Mar 2020, M. Clarke
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
        self.induced_power_factor     = 1.48  #accounts for interference effects
        self.profile_drag_coefficient = .03        
        self.tag                      = 'Propeller' 
    
    def spin(self,conditions):
        """Analyzes a propeller given geometry and operating conditions.

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
        beta_0 = self.twist_distribution
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
        BB     = B*B    
        BBB    = BB*B
    
        try:
            pitch_command = conditions.propulsion.pitch_command
            total_blade_pitch = beta_0 + pitch_command   
        except:
            total_blade_pitch = beta_0 
    
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
            lambda_i      = ua/(omega*R)
        else:              
            ut       = 0.0  
    
        #Things that don't change with iteration
        N        = len(c) # Number of stations     
        ctrl_pts = len(Vv)  
            
        if  a_pol != None and a_loc != None:
            airfoil_polars = Data() 
            # check dimension of section
            if len(a_loc) != N:
                raise AssertionError('Dimension of airfoil sections must be equal to number of stations on rotor')
            # compute airfoil polars for airfoils 
            airfoil_polars  = compute_airfoil_polars(self, a_geo, a_pol) 
            airfoil_cl_surs = airfoil_polars.lift_coefficient_surrogates
            airfoil_cd_surs = airfoil_polars.drag_coefficient_surrogates 
    
        # set up non dimensional radial distribution 
        if self.radius_distribution is None:
            chi0    = Rh/R   # Where the rotor blade actually starts
            chi     = np.linspace(chi0,1,N+1)  # Vector of nondimensional radii
            chi     = chi[0:N]
    
        else:
            chi = self.radius_distribution
    
        omega = np.abs(omega)        
        r_dim = chi*R                        # Radial coordinate 
        pi    = np.pi
        A     = pi*(R**2) 
        n     = omega/(2.*pi)                # Cycles per second  
        nu    = mu/rho    
        
        # blade area 
        blade_area   = sp.integrate.cumtrapz(B*c, r_dim-r_dim[0])
    
        # solidity 
        sigma        = blade_area[-1]/(pi*r_dim[-1]**2)                  # (page 28 Leishman)        
    
        # compute lambda and mu 
        lambda_tot   = (np.atleast_2d(V_inf[:,0]).T + ua)/(omega*R)       # inflow advance ratio (page 30 Leishman)
        mu_prop      = (np.atleast_2d(V_inf[:,2]).T) /(omega*R)           # rotor advance ratio  (page 30 Leishman)
        lambda_c     = (np.atleast_2d(V_inf[:,0]).T)/(omega*R)            # normal velocity ratio (page 30 Leishman)
        lambda_i     = ua/(omega*R)                                       # induced inflow ratio  (page 30 Leishman)
    
        # wake skew angle 
        X            = np.arctan(mu_prop/lambda_tot)
    
        # blade flap rate and sweep(cone) angle 
        beta_blade_dot = 0  # currently no flaping 
        beta_blade     = 0  # currently no coning            
    
        # azimuth distribution 
        psi          = np.linspace(0,2*pi,N)
        psi_2d       = np.tile(np.atleast_2d(psi).T,(1,N))
        psi_2d       = np.repeat(psi_2d[np.newaxis, :, :], ctrl_pts, axis=0)  
    
        # 2 dimensiona radial distribution non dimensionalized
        chi_2d       = np.tile(chi ,(N,1))            
        chi_2d       = np.repeat(chi_2d[ np.newaxis,:, :], ctrl_pts, axis=0) 
         
        r_dim_2d     = np.tile(r_dim ,(N,1))  
        r_dim_2d     = np.repeat(r_dim_2d[ np.newaxis,:, :], ctrl_pts, axis=0)  
    
        # Momentum theory approximation of inflow for BET if the advance ratio is large
        if conditions.use_Blade_Element_Theory :  
            '''Blade element theory (BET) assumes that each blade section acts as a two-dimensional
                airfoil for which the influence of the rotor wake consists entirely of an induced 
                velocity at the section. Two-dimensional airfoil characteristics can then be used
                to evaluate the section loads in terms of the blade motion and aerodynamic 
                environment at that section alone. The induced velocity can be obtained by various
                means: momentum theory, vortex theory, or nonuniform inflow calculations.
    
                Leishman pg 165'''     
    
            # 2-D chord distribution 
            chord     = np.tile(c,(N,1))  
            chord_2d  = np.repeat(chord[np.newaxis,:, :], ctrl_pts, axis=0)
    
            # 2-D blade twist distribution 
            theta_2d  = np.tile(total_blade_pitch,(N ,1))
            theta_2d  = np.repeat(theta_2d[np.newaxis,:, :], ctrl_pts, axis=0)    
    
            # 2-D inflow ratio 
            mu_2d     = np.tile(np.atleast_2d(mu_prop),(1,N))
            mu_2d     = np.repeat(mu_2d[:, np.newaxis,  :], N, axis=1)         
    
            # 2-D inflow ratio 
            lambda_2d = np.tile(np.atleast_2d(lambda_tot),(1,N))
            lambda_2d = np.repeat(lambda_2d[:, np.newaxis,  :], N, axis=1)       
    
            # wake skew angle 
            X     = np.arctan(mu_2d/lambda_2d) # wake skew angle (Section 5.2 page 133 Leishman) 
            kx_2d = np.tan(X/2)             # slope of downwash at center of rotor disk for forward flight eqn 5.42 (page 136 Leishman)        
            #kx_2d   = (4/3)*((1.8*mu_2d**2)*np.sqrt(1 + (lambda_2d/mu_2d)**2)  - lambda_2d/mu_2d)  # eqn 5.43 (page 136 Leishman)
            ky_2d = -2*mu_2d                # eqn 5.44 (page 136 Leishman) 
    
            lambda_i_2d = np.tile(np.atleast_2d(lambda_i),(1,N))
            lambda_i_2d = np.repeat(lambda_i_2d[:, np.newaxis,  :], N, axis=1) 
    
            lambda_c_2d = np.tile(np.atleast_2d(lambda_c),(1,N))
            lambda_c_2d = np.repeat(lambda_c_2d[:, np.newaxis,  :], N, axis=1)         
    
            # motification to initial radial inflow distribution  
            lambda_i_2d = lambda_i_2d*(1 + kx_2d*chi_2d*np.cos(psi_2d) + ky_2d*chi_2d*np.sin(psi_2d) )  # eqn 5.41 (page 136 Leishman)  
            lambda_2d   = lambda_c_2d + lambda_i_2d
    
            omega_2d   = np.tile(np.atleast_2d(omega),(1,N))
            omega_2d   = np.repeat(omega[:, np.newaxis,  :], N, axis=1)   
            Va_ind_2d  = lambda_i_2d*(omega_2d*R)     
            Vt_ind_2d  = np.zeros_like(Va_ind_2d)            
            
            # axial, tangential and radial components of local blade flow [multiplied by omega*R to dimensionalize] 
            omega_R_2d  = np.tile(np.atleast_2d(omega*R),(1,N))
            omega_R_2d  = np.repeat(omega_R_2d[:, np.newaxis,  :], N, axis=1)  
            Vt_2d       = omega_R_2d * (chi_2d  + mu_2d*np.sin(psi_2d))                                        # velocity tangential to the disk plane, positive toward the trailing edge eqn 6.34 pg 165           
            Vr_2d       = omega_R_2d * (mu_2d*np.cos(psi_2d))                                                # radial velocity , positive outward   eqn 6.35 pg 165                 
            Va_2d       = omega_R_2d * (lambda_2d + chi_2d *beta_blade_dot + beta_blade*mu_2d*np.cos(psi_2d))  # velocity perpendicular to the disk plane, positive downward  eqn 6.36 pg 166  
    
            # local total velocity 
            U_2d   = np.sqrt(Vt_2d**2 + Va_2d**2) # (page 165 Leishman)
    
            # blade incident angle 
            phi_2d = np.arctan(Va_2d/Vt_2d)     # (page 166 Leishman)
    
            # local blade angle of attack
            alpha  = theta_2d - phi_2d  # (page 166 Leishman)
    
            # Scale for Mach, this is Karmen_Tsien 
            a_2d  = np.tile(np.atleast_2d(a),(1,N))
            a_2d  = np.repeat(a_2d[:, np.newaxis,  :], N, axis=1)  
            
            # local mach number
            Ma         = (U_2d)/a_2d   
            
            # Estimate Cl max  
            nu_2d      = np.tile(np.atleast_2d(nu),(1,N))
            nu_2d      = np.repeat(nu_2d[:, np.newaxis,  :], N, axis=1)   
            Re         = (U_2d*chord_2d)/nu_2d 
            Cl_max_ref = -0.0009*tc**3 + 0.0217*tc**2 - 0.0442*tc + 0.7005
            Re_ref     = 9.*10**6      
            Cl1maxp    = Cl_max_ref * ( Re / Re_ref ) **0.1   #THIS IS INCORRECT
     
            
            # Compute blade Cl and Cd distribution from the airfoil data if provided else use thin airfoil theory 
            if  a_pol != None and a_loc != None:  
                Cl    = np.zeros((ctrl_pts,N,N))              
                Cdval = np.zeros((ctrl_pts,N,N))                 
                for ii in range(ctrl_pts):
                    for jj in range(N):                 
                        Cl[ii,:,jj]    = airfoil_cl_surs[a_geo[a_loc[jj]]](Re[ii,:,jj],alpha[ii,:,jj],grid=False)  
                        Cdval[ii,:,jj] = airfoil_cd_surs[a_geo[a_loc[jj]]](Re[ii,:,jj],alpha[ii,:,jj],grid=False)  
            else:
                # If not airfoil polar provided, use 2*pi as lift curve slope
                Cl = 2.*pi*alpha
    
                # By 90 deg, it's totally stalled.
                Cl[Cl>Cl1maxp]  = Cl1maxp[Cl>Cl1maxp]  
                Cl[alpha>=pi/2] = 0.  
                Cl[Ma[:,:]<1.]  = Cl[Ma[:,:]<1.]/((1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5+((Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])/(1+(1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5))*Cl[Ma<1.]/2)
                
                # If the blade segments are supersonic, don't scale
                Cl[Ma[:,:]>=1.] = Cl[Ma[:,:]>=1.]   
            
                #There is also RE scaling
                #This is an atrocious fit of DAE51 data at RE=50k for Cd
                Cdval              = (0.108*(Cl*Cl*Cl*Cl)-0.2612*(Cl*Cl*Cl)+0.181*(Cl*Cl)-0.0139*Cl+0.0278)*((50000./Re)**0.2)
                Cdval[alpha>=pi/2] = 2.    
    
            #More Cd scaling from Mach from AA241ab notes for turbulent skin friction 
            T_2d    = np.tile(np.atleast_2d(T),(1,N))
            T_2d    = np.repeat(T_2d[:, np.newaxis,  :], N, axis=1)     
            Tw_Tinf = 1. + 1.78*(Ma*Ma)
            Tp_Tinf = 1. + 0.035*(Ma*Ma) + 0.45*(Tw_Tinf-1.)
            Tp      = (Tp_Tinf)*T_2d
            Rp_Rinf = (Tp_Tinf**2.5)*(Tp+110.4)/(T_2d+110.4) 
            Cd      = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval  
    
            # local blade lift and drag  
            rho_2d = np.tile(np.atleast_2d(rho),(1,N))
            rho_2d = np.repeat(rho_2d[:, np.newaxis,  :], N, axis=1)    
            dL     = 0.5 * rho_2d * U_2d**2 * chord_2d * Cl # eqn 6.37 (page 167 Leishman) 
            dD     = 0.5 * rho_2d * U_2d**2 * chord_2d * Cd # eqn 6.38 (page 167 Leishman) 
    
            # application of tip loss factor 
            tip_loss_factor          = 0.97 # (page 67 and  Leishman) make a property of the rotor
            dL[chi_2d>tip_loss_factor] = 0    # (page 63 & 90 and  Leishman) 
    
            # normal and tangential forces  
            dFz  = dL*np.cos(phi_2d) - dD*np.sin(phi_2d) # eqn 6.39 (page 167 Leishman) 
            dFx  = dL*np.sin(phi_2d) - dD*np.cos(phi_2d) # eqn 6.40 (page 167 Leishman)
    
            # average thrust and torque over aximuth
            deltar                  = chi_2d[:,:,1]-chi_2d[:,:,0]  
            deltar_2d               = np.repeat(deltar[:, np.newaxis,  :], N, axis=1)  
            blade_T_distribution    = np.mean((dFz*deltar_2d), axis = 1)
            blade_Q_distribution    = np.mean((dFx*chi_2d*deltar_2d), axis = 1)
            thrust                  = np.atleast_2d((B * np.sum(blade_T_distribution, axis = 1))).T 
            torque                  = np.atleast_2d((B * np.sum(blade_Q_distribution, axis = 1))).T 
            blade_T_distribution_2d = dFz*deltar_2d
            blade_Q_distribution_2d = dFx*chi_2d*deltar_2d  
            
            blade_dT_dR = np.zeros_like(deltar)
            blade_dT_dr = np.zeros_like(deltar)
            blade_dQ_dR = np.zeros_like(deltar)
            blade_dQ_dr = np.zeros_like(deltar)
             
            blade_Gamma_2d  = 0.5*U_2d*chord_2d*Cl       
        
            for i in range(len(Vv)):
                blade_dT_dR[i,:] = np.gradient(blade_T_distribution[i], deltar[i,0]*R) 
                blade_dT_dr[i,:] = np.gradient(blade_T_distribution[i], deltar[i,0])
                blade_dQ_dR[i,:] = np.gradient(blade_Q_distribution[i], deltar[i,0]*R)
                blade_dQ_dr[i,:] = np.gradient(blade_Q_distribution[i], deltar[i,0])  
                
    
        # Blade Element Momentum Theory : large angle approximation
        else:   
            #Things that will change with iteration
            size   = (len(a),N) 
            tol    = 1e-5
            omegar = np.outer(omega,r_dim)
            Ua     = np.outer((V + ua),np.ones_like(r_dim))
            Ut     = omegar - ut
            U      = np.sqrt(Ua*Ua + Ut*Ut)
            pi2    = pi*pi
            beta   = total_blade_pitch
            
            #Setup a Newton iteration
            PSI    = np.ones(size)
            psiold = np.zeros(size)
            diff   = 1.
        
            ii = 0
            broke = False   
            while (diff>tol):
                sin_psi = np.sin(PSI)
                cos_psi = np.cos(PSI)
                Wa      = 0.5*Ua + 0.5*U*sin_psi
                Wt      = 0.5*Ut + 0.5*U*cos_psi   
                va      = Wa - Ua
                vt      = Ut - Wt
                alpha   = beta - np.arctan2(Wa,Wt)
                W       = (Wa*Wa + Wt*Wt)**0.5
                Ma      = (W)/a #a is the speed of sound 
        
                lamdaw = r_dim*Wa/(R*Wt)
        
                # Limiter to keep from Nan-ing
                lamdaw[lamdaw<0.] = 0.
        
                f            = (B/2.)*(1.-r_dim/R)/lamdaw
                piece        = np.exp(-f)
                arccos_piece = np.arccos(piece)
                F            = 2.*arccos_piece/pi
                Gamma        = vt*(4.*pi*r_dim/B)*F*(1.+(4.*lamdaw*R/(pi*B*r_dim))*(4.*lamdaw*R/(pi*B*r_dim)))**0.5
        
                # Estimate Cl max
                Re         = (W*c)/nu 
                Cl_max_ref = -0.0009*tc**3 + 0.0217*tc**2 - 0.0442*tc + 0.7005
                Re_ref     = 9.*10**6      
                Cl1maxp    = Cl_max_ref * ( Re / Re_ref ) **0.1
        

                # Compute blade Cl and Cd distribution from the airfoil data if provided else use thin airfoil theory   
                if  a_pol != None and a_loc != None: 
                    Cl    = np.zeros((ctrl_pts,N))              
                    Cdval = np.zeros((ctrl_pts,N)) 
                    for jj in range(N):                 
                        Cl[:,jj]    = airfoil_cl_surs[a_geo[a_loc[jj]]](Re[:,jj],alpha[:,jj],grid=False)  
                        Cdval[:,jj] = airfoil_cd_surs[a_geo[a_loc[jj]]](Re[:,jj],alpha[:,jj],grid=False)    
                        
                else:
                    # If not airfoil polar provided, use 2*pi as lift curve slope
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
                    Cdval              = (0.108*(Cl*Cl*Cl*Cl)-0.2612*(Cl*Cl*Cl)+0.181*(Cl*Cl)-0.0139*Cl+0.0278)*((50000./Re)**0.2)
                    Cdval[alpha>=pi/2] = 2.         
    
                #More Cd scaling from Mach from AA241ab notes for turbulent skin friction
                Tw_Tinf = 1. + 1.78*(Ma*Ma)
                Tp_Tinf = 1. + 0.035*(Ma*Ma) + 0.45*(Tw_Tinf-1.)
                Tp      = (Tp_Tinf)*T
                Rp_Rinf = (Tp_Tinf**2.5)*(Tp+110.4)/(T+110.4)
        
                Cd = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval 	
                
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
        
                dR_dpsi = ((4.*U*r_dim*arccos_piece*sin_psi*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5))/B - 
                               (pi*U*(Ua*cos_psi - Ut*sin_psi)*(beta - np.arctan((Wa+Wa)/(Wt+Wt))))/(2.*(f_wt_2 + f_wa_2)**(0.5))
                               + (pi*U*(f_wt_2 +f_wa_2)**(0.5)*(U + Utcospsi  +  Uasinpsi))/(2.*(f_wa_2/(f_wt_2) + 1.)*utpUcospsi2)
                               - (4.*U*piece*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5)*(R - r_dim)*(Ut/2. - 
                                                                                                      (Ucospsi)/2.)*(U + Utcospsi + Uasinpsi ))/(f_wa_2*(1. - np.exp(-(B*(Wt+Wt)*(R - 
                                                                                                          r_dim))/(r_dim*(Wa+Wa))))**(0.5)) + (128.*U*r_dim*arccos_piece*(Wa+Wa)*(Ut/2. - (Ucospsi)/2.)*(U + 
                                                                                                                  Utcospsi  + Uasinpsi ))/(BBB*pi2*utpUcospsi*utpUcospsi2*((16.*f_wa_2)/(BB*pi2*f_wt_2) + 1.)**(0.5))) 
        
                dR_dpsi[np.isnan(dR_dpsi)] = 0.1
        
                dpsi   = -Rsquiggly/dR_dpsi
                PSI    = PSI + dpsi
                diff   = np.max(abs(psiold-PSI))
                psiold = PSI
        
                # If its really not going to converge
                if np.any(PSI>(pi*85.0/180.)) and np.any(dpsi>0.0):
                    break
        
                ii+=1
        
                if ii>2000:
                    broke = True
                    break
    
            epsilon  = Cd/Cl
            epsilon[epsilon==np.inf] = 10. 
            deltar   = (r_dim[1]-r_dim[0]) 
                        
            blade_T_distribution = rho*(Gamma*(Wt-epsilon*Wa))*deltar 
            blade_Q_distribution = rho*(Gamma*(Wa+epsilon*Wt)*r_dim)*deltar 
            thrust               = rho*B*(np.sum(Gamma*(Wt-epsilon*Wa)*deltar,axis=1)[:,None])
            torque               = rho*B*np.sum(Gamma*(Wa+epsilon*Wt)*r_dim*deltar,axis=1)[:,None]
    
            Va_2d     = np.repeat(Wa.T[ : , np.newaxis , :], N, axis=1).T
            Vt_2d     = np.repeat(Wt.T[ : , np.newaxis , :], N, axis=1).T
            Vt_ind_2d = np.repeat(va.T[ : , np.newaxis , :], N, axis=1).T
            Va_ind_2d = np.repeat(vt.T[ : , np.newaxis , :], N, axis=1).T
            blade_T_distribution_2d = np.repeat(blade_T_distribution.T[ np.newaxis,:  , :], N, axis=0).T 
            blade_Q_distribution_2d = np.repeat(blade_Q_distribution.T[ np.newaxis,:  , :], N, axis=0).T                 
             
            blade_Gamma_2d  = np.repeat(Gamma.T[ : , np.newaxis , :], N, axis=1).T
            blade_dT_dR     = rho*(Gamma*(Wt-epsilon*Wa))
            blade_dT_dr     = rho*(Gamma*(Wt-epsilon*Wa))*R
            blade_dQ_dR     = rho*(Gamma*(Wa+epsilon*Wt)*r_dim)
            blade_dQ_dr     = rho*(Gamma*(Wa+epsilon*Wt)*r_dim)*R
            
        psi_2d      = np.repeat(np.atleast_2d(psi).T[np.newaxis,: ,:], N, axis=0).T        
        D           = 2*R 
        Cq          = torque/(rho*A*R*(omega*R)**2)
        Ct          = thrust/(rho*A*(omega*R)**2)
        Ct[Ct<0]    = 0.     # prevent things from breaking
        kappa       = self.induced_power_factor 
        Cd0         = self.profile_drag_coefficient   
        Cp          = np.zeros_like(Ct)
        power       = np.zeros_like(Ct)      
    
        for i in range(len(Vv)):   
            if -1. <Vv[i][0] <1.: # vertical/axial flight
                Cp[i]       = (kappa*(Ct[i]**1.5)/(2**.5))+sigma*Cd0/8.
                power[i]    = Cp[i]*(rho[i]*(n[i]*n[i]*n[i])*(D*D*D*D*D))
                torque[i]   = power[i]/omega[i]  
            else:  
                power[i]    = torque[i]*omega[i]  
                Cp[i]       = power[i]/(rho[i]*(n[i]*n[i]*n[i])*(D*D*D*D*D)) 
    
        # torque coefficient 
        Cq = torque/(rho*(n*n)*(D*D*D*D)*R) 
    
        thrust[conditions.propulsion.throttle[:,0] <=0.0] = 0.0
        power[conditions.propulsion.throttle[:,0]  <=0.0] = 0.0 
        torque[conditions.propulsion.throttle[:,0] <=0.0] = 0.0 
    
        etap     = V*thrust/power     
    
        conditions.propulsion.etap = etap 
    
        # store data
        results_conditions                              = Data     
        outputs                                         = results_conditions(
                    num_blades                              = int(B),
                    propeller_radius                        = R,
                    propeller_diameter                      = D,
                    number_sections                         = N,
                    blade_radial_distribution_normalized    = chi,
                    blade_radial_distribution_normalized_2d = chi_2d,
                    blade_chord_distribution                = c,     
                    blade_twist_distribution                = beta_0,            
                    blade_radial_distribution               = r_dim,  
                    blade_radial_distribution_2d            = r_dim_2d,  
                    thrust_angle                            = theta,
                    speed_of_sound                          = conditions.freestream.speed_of_sound,
                    density                                 = conditions.freestream.density,
                    velocity                                = Vv, 
                    tangential_induced_velocity_2d          = Vt_ind_2d, 
                    axial_induced_velocity_2d               = Va_ind_2d,  
                    tangential_velocity_2d                  = Vt_2d, 
                    axial_velocity_2d                       = Va_2d, 
                    drag_coefficient                        = Cd,
                    lift_coefficient                        = Cl,       
                    omega                                   = omega,  
                    blade_Gamma_2d                          = blade_Gamma_2d,
                    blade_dT_dR                             = blade_dT_dR,
                    blade_dT_dr                             = blade_dT_dr,
                    thrust_distribution                     = blade_T_distribution, 
                    thrust_distribution_2d                  = blade_T_distribution_2d, 
                    thrust_per_blade                        = thrust/B, 
                    thrust_coefficient                      = Ct,   
                    azimuthal_distribution                  = psi, 
                    azimuthal_distribution_2d               = psi_2d ,
                    blade_dQ_dR                             = blade_dQ_dR ,
                    blade_dQ_dr                             = blade_dQ_dr ,
                    torque_distribution                     = blade_Q_distribution, 
                    torque_distribution_2d                  = blade_Q_distribution_2d, 
                    torque_per_blade                        = torque/B,   
                    torque_coefficient                      = Cq,   
                    power                                   = power,
                    power_coefficient                       = Cp, 
                    mid_chord_aligment                      = self.mid_chord_aligment     
            ) 
    
        return thrust, torque, power, Cp, outputs  , etap 
