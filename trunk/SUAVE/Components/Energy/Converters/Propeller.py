# Propeller.py
#
# Created:  Jun 2014, E. Botero
# Modified: Jan 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Core import Data
from SUAVE.Analyses import Results
import scipy.optimize as opt

from SUAVE.Methods.Geometry.Three_Dimensional \
     import angles_to_dcms, orientation_product, orientation_transpose

from warnings import warn

# ----------------------------------------------------------------------
#  Propeller Class
# ----------------------------------------------------------------------    
 
class Propeller(Energy_Component):
    
    def __defaults__(self):
        
        self.prop_attributes = Data
        self.prop_attributes.number_blades      = 0.0
        self.prop_attributes.tip_radius         = 0.0
        self.prop_attributes.hub_radius         = 0.0
        self.prop_attributes.twist_distribution = 0.0
        self.prop_attributes.chord_distribution = 0.0
        self.prop_attributes.mid_chord_aligment = 0.0
        self.thrust_angle                       = 0.0
        
    def spin(self,conditions):
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
        B      = self.prop_attributes.number_blades
        R      = self.prop_attributes.tip_radius
        Rh     = self.prop_attributes.hub_radius
        beta   = self.prop_attributes.twist_distribution
        c      = self.prop_attributes.chord_distribution
        omega1 = self.inputs.omega
        rho    = conditions.freestream.density[:,0,None]
        mu     = conditions.freestream.dynamic_viscosity[:,0,None]
        Vv     = conditions.frames.inertial.velocity_vector
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
        
        # Velocity transformed to the propulsor frame
        body2thrust   = np.array([[np.cos(theta), 0., np.sin(theta)],[0., 1., 0.], [-np.sin(theta), 0., np.cos(theta)]])
        T_body2thrust = orientation_transpose(np.ones_like(T_body2inertial[:])*body2thrust)
        V_thrust      = orientation_product(T_body2thrust,V_body)
        
        # Now just use the aligned velocity
        V = V_thrust[:,0,None]
        
        nu    = mu/rho
        tol   = 1e-6 # Convergence tolerance
        
        omega = omega1*1.0
        omega = np.abs(omega)
           
        ######
        # Enter airfoil data in a better way, there is currently Re and Ma scaling from DAE51 data
        ######

        #Things that don't change with iteration
        N       = len(c) # Number of stations
        chi0    = Rh/R   # Where the propeller blade actually starts
        chi     = np.linspace(chi0,1,N+1)  # Vector of nondimensional radii
        chi     = chi[0:N]
        lamda   = V/(omega*R)              # Speed ratio
        r       = chi*R                    # Radial coordinate
        pi      = np.pi
        pi2     = pi*pi
        x       = r*np.multiply(omega,1/V) # Nondimensional distance
        n       = omega/(2.*pi)            # Cycles per second
        J       = V/(2.*R*n)    
        sigma   = np.multiply(B*c,1./(2.*pi*r))          
    
        #I make the assumption that externally-induced velocity at the disk is zero
        #This can be easily changed if needed in the future:
        ua = 0.0
        ut = 0.0
        
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
        while (diff>tol):
            sin_psi = np.sin(psi)
            cos_psi = np.cos(psi)
            Wa      = 0.5*Ua + 0.5*U*sin_psi
            Wt      = 0.5*Ut + 0.5*U*cos_psi   
            #va     = Wa - Ua
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
            #Cl_max_ref = -0.0009*tc**3 + 0.0217*tc**2 - 0.0442*tc + 0.7005
            #Re_ref     = 9.*10**6      
            #Cl1maxp    = Cl_max_ref * ( Re / Re_ref ) **0.1
            
            # Ok, from the airfoil data, given Re, Ma, alpha we need to find Cl
            Cl = 2.*pi*alpha
            
            # By 90 deg, it's totally stalled.
            #Cl[Cl>Cl1maxp]  = Cl1maxp[Cl>Cl1maxp]
            Cl[alpha>=pi/2] = 0.
            
            
            ## Scale for Mach, this is Karmen_Tsien
            #Cl[Ma[:,:]<1.] = Cl[Ma[:,:]<1.]/((1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5+((Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])/(1+(1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5))*Cl[Ma<1.]/2)
            
            ## If the blade segments are supersonic, don't scale
            #Cl[Ma[:,:]>=1.] = Cl[Ma[:,:]>=1.] 
            
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
                print 'broke'
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
        power    = torque*omega       
       
        D        = 2*R
        Cp       = power/(rho*(n*n*n)*(D*D*D*D*D))

        thrust[conditions.propulsion.throttle[:,0] <=0.0] = 0.0
        power[conditions.propulsion.throttle[:,0]  <=0.0] = 0.0
        
        thrust[omega1<0.0] = - thrust[omega1<0.0]

        etap     = V*thrust/power     
        
        conditions.propulsion.etap = etap
        
        # store data
        results_conditions = Results       
        conditions.propulsion.acoustic_outputs = results_conditions(
            number_sections    = N,
            r0                 = r,
            airfoil_chord      = c,
            blades_number      = B,
            propeller_diameter = D,
            drag_coefficient   = Cd,
            lift_coefficient   = Cl,
            omega              = omega,
            velocity           = V,
            thrust             = thrust,
            power              = power,
            mid_chord_aligment = self.prop_attributes.mid_chord_aligment
        )
        
        
        return thrust, torque, power, Cp
    
    

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
        B       = self.prop_attributes.number_blades
        R       = self.prop_attributes.tip_radius
        Rh      = self.prop_attributes.hub_radius
        beta_in = self.prop_attributes.twist_distribution
        c       = self.prop_attributes.chord_distribution
        omega1  = self.inputs.omega
        rho     = conditions.freestream.density[:,0,None]
        mu      = conditions.freestream.dynamic_viscosity[:,0,None]
        Vv      = conditions.frames.inertial.velocity_vector
        a       = conditions.freestream.speed_of_sound[:,0,None]
        T       = conditions.freestream.temperature[:,0,None]
        theta   = self.thrust_angle
        tc      = .12 # Thickness to chord
        beta_c  = conditions.propulsion.pitch_command
        
        beta   = beta_in + beta_c
        
        BB     = B*B
        BBB    = BB*B
            
        # Velocity in the Body frame
        T_body2inertial = conditions.frames.body.transform_to_inertial
        T_inertial2body = orientation_transpose(T_body2inertial)
        V_body = orientation_product(T_inertial2body,Vv)
        
        # Velocity transformed to the propulsor frame
        body2thrust   = np.array([[np.cos(theta), 0., np.sin(theta)],[0., 1., 0.], [-np.sin(theta), 0., np.cos(theta)]])
        T_body2thrust = orientation_transpose(np.ones_like(T_body2inertial[:])*body2thrust)
        V_thrust      = orientation_product(T_body2thrust,V_body)
        
        # Now just use the aligned velocity
        V = V_thrust[:,0,None]
        
        nu    = mu/rho
        tol   = 1e-6 # Convergence tolerance
        
        omega = omega1*1.0
        omega = np.abs(omega)
           
        ######
        # Enter airfoil data in a better way, there is currently Re and Ma scaling from DAE51 data
        ######

        #Things that don't change with iteration
        N       = len(c) # Number of stations
        chi0    = Rh/R   # Where the propeller blade actually starts
        chi     = np.linspace(chi0,1,N+1)  # Vector of nondimensional radii
        chi     = chi[0:N]
        lamda   = V/(omega*R)              # Speed ratio
        r       = chi*R                    # Radial coordinate
        pi      = np.pi
        pi2     = pi*pi
        x       = r*np.multiply(omega,1/V) # Nondimensional distance
        n       = omega/(2.*pi)            # Cycles per second
        J       = V/(2.*R*n)    
        sigma   = np.multiply(B*c,1./(2.*pi*r))          
    
        #I make the assumption that externally-induced velocity at the disk is zero
        #This can be easily changed if needed in the future:
        ua = 0.0
        ut = 0.0
        
        omegar = np.outer(omega,r)
        Ua = np.outer((V + ua),np.ones_like(r))
        Ut = omegar - ut
        U  = np.sqrt(Ua*Ua + Ut*Ut)
        
        #Things that will change with iteration
        size = (len(a),N)
    
        #Setup a Newton iteration
        psi    = np.ones(size)*0.5
        psiold = np.zeros(size)
        diff   = 1.
        
        ii = 0
        while (diff>tol):
            sin_psi = np.sin(psi)
            cos_psi = np.cos(psi)
            Wa      = 0.5*Ua + 0.5*U*sin_psi
            Wt      = 0.5*Ut + 0.5*U*cos_psi   
            #va     = Wa - Ua
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
            #Cl_max_ref = -0.0009*tc**3 + 0.0217*tc**2 - 0.0442*tc + 0.7005
            #Re_ref     = 9.*10**6      
            #Cl1maxp    = Cl_max_ref * ( Re / Re_ref ) **0.1
            
            # Ok, from the airfoil data, given Re, Ma, alpha we need to find Cl
            Cl = 2.*pi*alpha
            
            # By 90 deg, it's totally stalled.
            #Cl[Cl>Cl1maxp]  = Cl1maxp[Cl>Cl1maxp]
            Cl[alpha>=pi/2] = 0.
            
            
            ## Scale for Mach, this is Karmen_Tsien
            #Cl[Ma[:,:]<1.] = Cl[Ma[:,:]<1.]/((1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5+((Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])/(1+(1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5))*Cl[Ma<1.]/2)
            
            ## If the blade segments are supersonic, don't scale
            #Cl[Ma[:,:]>=1.] = Cl[Ma[:,:]>=1.] 
            
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
                print 'broke'
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
        power    = torque*omega       
       
        D        = 2*R
        Cp       = power/(rho*(n*n*n)*(D*D*D*D*D))

        thrust[conditions.propulsion.throttle[:,0] <=0.0] = 0.0
        power[conditions.propulsion.throttle[:,0]  <=0.0] = 0.0
        
        thrust[omega1<0.0] = - thrust[omega1<0.0]

        etap     = V*thrust/power     
        
        conditions.propulsion.etap = etap
        
        # store data
        results_conditions = Results       
        conditions.propulsion.acoustic_outputs = results_conditions(
            number_sections = N,
            r0 = r,
            airfoil_chord = c,
            blades_number = B,
            propeller_diameter = D,
            drag_coefficient = Cd,
            lift_coefficient = Cl,
            rpm = omega,
            velocity = V,
            thrust = thrust,
            hp = power,
        )
        
        
        return thrust, torque, power, Cp
