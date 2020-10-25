## @ingroup Components-Energy-Converters
# Rotor.py
#
# Created:  Jun 2014, E. Botero
# Modified: Jan 2016, T. MacDonald
#           Feb 2019, M. Vegh            
#           Mar 2020, M. Clarke
#           Sep 2020, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Core import Data
from SUAVE.Methods.Geometry.Three_Dimensional \
     import  orientation_product, orientation_transpose

# package imports
import numpy as np

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
        
        self.number_blades             = 0.0
        self.tip_radius                = 0.0
        self.hub_radius                = 0.0
        self.twist_distribution        = 0.0
        self.chord_distribution        = 0.0
        self.mid_chord_aligment        = 0.0
        self.blade_solidity            = 0.0
        self.thrust_angle              = 0.0
        self.pitch_command             = 0.0
        self.design_power              = None
        self.design_thrust             = None        
        self.induced_hover_velocity    = None
        self.airfoil_geometry          = None
        self.airfoil_polars            = None
        self.airfoil_polar_stations    = None 
        self.radius_distribution       = None
        self.rotation                  = None
        self.ducted                    = False
        self.number_azimuthal_stations = 24
        self.induced_power_factor      = 1.48  #accounts for interference effects
        self.profile_drag_coefficient  = .03        
        self.tag                       = 'Rotor'


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
        a       = conditions.freestream.speed_of_sound[:,0,None]
        T       = conditions.freestream.temperature[:,0,None]
        pitch_c = self.pitch_command
        theta   = self.thrust_angle 
        Na      = self.number_azimuthal_stations 
        BB      = B*B    
        BBB     = BB*B
    
        # calculate total blade pitch
        total_blade_pitch = beta_0 + pitch_c  
            
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
        pi2            = pi*pi       
        n              = omega/(2.*pi)                      # Cycles per second  
        nu             = mu/rho         
    
        # azimuth distribution 
        psi            = np.linspace(0,2*pi,Na+1)[:-1]
        psi_2d         = np.tile(np.atleast_2d(psi).T,(1,Nr))
        psi_2d         = np.repeat(psi_2d[np.newaxis, :, :], ctrl_pts, axis=0)   
        azimuth_2d     = np.repeat(np.atleast_2d(psi).T[np.newaxis,: ,:], Na, axis=0).T  
        
        # 2 dimensiona radial distribution non dimensionalized
        chi_2d         = np.tile(chi ,(Na,1))            
        chi_2d         = np.repeat(chi_2d[ np.newaxis,:, :], ctrl_pts, axis=0) 
                       
        r_dim_2d       = np.tile(r ,(Na,1))  
        r_dim_2d       = np.repeat(r_dim_2d[ np.newaxis,:, :], ctrl_pts, axis=0)  
    
        # Things that will change with iteration
        size   = (len(a),Nr)
        omegar = np.outer(omega,r)
        Ua     = np.outer((V + ua),np.ones_like(r))
        Ut     = omegar - ut
        U      = np.sqrt(Ua*Ua + Ut*Ut) 
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
                print("Propeller BEMT did not converge to a solution")
                break
        
            ii+=1 
            if ii>10000:
                broke = True
                print("Propeller BEMT did not converge to a solution")
                break
    
        # More Cd scaling from Mach from AA241ab notes for turbulent skin friction
        Tw_Tinf     = 1. + 1.78*(Ma*Ma)
        Tp_Tinf     = 1. + 0.035*(Ma*Ma) + 0.45*(Tw_Tinf-1.)
        Tp          = (Tp_Tinf)*T
        Rp_Rinf     = (Tp_Tinf**2.5)*(Tp+110.4)/(T+110.4) 
        Cd          = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval  
        
        epsilon                  = Cd/Cl
        epsilon[epsilon==np.inf] = 10. 
        deltar                   = (r[1]-r[0])  
        
        blade_T_distribution     = rho*(Gamma*(Wt-epsilon*Wa))*deltar 
        blade_Q_distribution     = rho*(Gamma*(Wa+epsilon*Wt)*r)*deltar 
        thrust                   = rho*B*(np.sum(Gamma*(Wt-epsilon*Wa)*deltar,axis=1)[:,None])
        torque                   = rho*B*np.sum(Gamma*(Wa+epsilon*Wt)*r*deltar,axis=1)[:,None]  
        power                    = omega*torque  
        Va_2d                    = np.repeat(Wa.T[ : , np.newaxis , :], Na, axis=1).T
        Vt_2d                    = np.repeat(Wt.T[ : , np.newaxis , :], Na, axis=1).T
        Va_ind_2d                = np.repeat(va.T[ : , np.newaxis , :], Na, axis=1).T
        Vt_ind_2d                = np.repeat(vt.T[ : , np.newaxis , :], Na, axis=1).T
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
                
        # calculate coefficients 
        D        = 2*R 
        Cq       = torque/(rho*(n*n)*(D*D*D*D*D)) 
        Ct       = thrust/(rho*(n*n)*(D*D*D*D))
        Cp       = power/(rho*(n*n*n)*(D*D*D*D*D))  # correct 
        etap     = V*thrust/power # efficiency    

        # prevent things from breaking 
        Cq[Cq<0]                                           = 0.  
        Ct[Ct<0]                                           = 0.  
        Cp[Cp<0]                                           = 0.  
        thrust[conditions.propulsion.throttle[:,0] <=0.0]  = 0.0
        power[conditions.propulsion.throttle[:,0]  <=0.0]  = 0.0 
        torque[conditions.propulsion.throttle[:,0]  <=0.0] = 0.0
        thrust[omega<0.0]                                  = - thrust[omega<0.0]  
        thrust[omega==0.0]                                 = 0.0
        power[omega==0.0]                                  = 0.0
        torque[omega==0.0]                                 = 0.0
        Ct[omega==0.0]                                     = 0.0
        Cp[omega==0.0]                                     = 0.0 
        etap[omega==0.0]                                   = 0.0 
        
        # assign efficiency to network
        conditions.propulsion.etap = etap   
                
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
                    disc_azimuthal_distribution       = azimuth_2d,
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
