## @ingroup Components-Energy-Converters
# Two_Dimensional_Inlet.py
#
# Created:  July 2019, M. Dethy

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE

# python imports
from warnings import warn

# package imports
import numpy as np

from SUAVE.Core import Data, Units
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Aerodynamics.Common.Gas_Dynamics import Oblique_Shock, Isentropic

# ----------------------------------------------------------------------
#  Two Dimensional Inlet Component
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
class Two_Dimensional_Inlet(Energy_Component):
    """This is a two dimensional inlet component intended for use in compression.
    Calling this class calls the compute function.

    Source:
    https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/
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
        # setting the default values
        self.tag = '2D_inlet'
        self.spillage_fraction               = 0.0 # fraction of mass flow lost to spillage
        self.areas                           = Data()
        self.areas.engine_face               = 0.0 
        self.areas.ramp_area                 = 0.0
        self.areas.inlet_capture             = 0.0
        self.areas.drag_direct_projection    = 0.0
        self.angles                          = Data()
        self.angles.ramp_angle               = 0.0 # should be in radians
        self.inputs.stagnation_temperature   = np.array([0.0])
        self.inputs.stagnation_pressure      = np.array([0.0])
        self.outputs.stagnation_temperature  = np.array([0.0])
        self.outputs.stagnation_pressure     = np.array([0.0])
        self.outputs.stagnation_enthalpy     = np.array([0.0])
        self.inlet_drag_calc                 = np.array([0.0])
        self.compute_pressure_ratios         = False

    def compute(self, conditions):
        
        """ This computes the output values from the input values according to
        equations from the source.

        Assumptions:
        Constant polytropic efficiency and pressure ratio
        Adiabatic

        Source:
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/

        Inputs:
        conditions.freestream.
          isentropic_expansion_factor         [-]
          specific_heat_at_constant_pressure  [J/(kg K)]
          pressure                            [Pa]
          gas_specific_constant               [J/(kg K)]
        self.inputs.
          stagnation_temperature              [K]
          stagnation_pressure                 [Pa]

        Outputs:
        self.outputs.
          stagnation_temperature              [K]
          stagnation_pressure                 [Pa]
          stagnation_enthalpy                 [J/kg]
          mach_number                         [-]
          static_temperature                  [K]
          static_enthalpy                     [J/kg]
          velocity                            [m/s]

        Properties Used:
        self.
          pressure_ratio                      [-]
          polytropic_efficiency               [-]
          pressure_recovery                   [-]
        """

        # unpack segment flow conditions
        gamma = conditions.freestream.isentropic_expansion_factor
        Cp    = conditions.freestream.specific_heat_at_constant_pressure
        R     = conditions.freestream.gas_specific_constant
        P0    = conditions.freestream.pressure
        M0    = np.atleast_2d(conditions.freestream.mach_number)
        A0    = conditions.freestream.area_initial_streamtube
        
        A0h = np.ones_like(M0)

        # unpack from inputs
        Tt0 = self.inputs.stagnation_temperature
        Pt0 = self.inputs.stagnation_pressure

        # unpack from self
        A2    = self.areas.engine_face # engine face area
        AC    = self.areas.inlet_capture # inlet capture area
        A1    = self.areas.ramp_area
        theta = self.angles.ramp_angle/Units.deg # incoming angle for the shock converted to degrees
        Ks    = self.spillage_fraction # fraction of incoming mass lost to spillage
        
        # Compute the mass flow rate into the engine
        T0              = Isentropic.isentropic_relations(M0, gamma)[0]*Tt0
        v0              = np.sqrt(gamma*R*T0)*M0

        f_M0            = Isentropic.isentropic_relations(M0, gamma)[-1]
        
        i_sub           = M0 <= 1.0
        i_sup           = M0 > 1.0

        # initializing the arrays
        Tt_out  = Tt0
        ht_out  = Cp*Tt0
        Pt_out  = np.ones_like(M0)
        Mach    = np.ones_like(M0)
        T_out   = np.ones_like(M0)
        f_M2    = np.ones_like(M0)
        
    
        M1      = np.ones_like(M0)
        Pr1     = np.ones_like(M0)
        Tr1     = np.ones_like(M0)
        Ptr1    = np.ones_like(M0)  
        Pt1     = np.ones_like(M0)
        f_M1    = np.ones_like(M0)
        beta    = np.ones_like(M0)
        
        # Subsonic flow into inlet (Assuming no normal shock at throat)
        Pt_out[i_sub]   = Pt0[i_sub]
        f_M2[i_sub]     = (1-Ks)*(f_M0[i_sub] * A0[i_sub])/A2
        Mach[i_sub]     = Isentropic.get_m(f_M2[i_sub], gamma[i_sub], 1)
        T_out[i_sub]    = Isentropic.isentropic_relations(Mach[i_sub], gamma[i_sub])[0]*Tt_out[i_sub]     
        
        # Supersonic flow into inlet (strong oblique shock on impact with ramp)
        beta[i_sup]                                     = Oblique_Shock.theta_beta_mach(M0[i_sup],gamma[i_sup],theta*Units.deg,1)
        beta[i_sup]                                     = np.nan_to_num(beta[i_sup], nan=np.pi/2)
        M1[i_sup], Pr1[i_sup], Tr1[i_sup], Ptr1[i_sup]  = Oblique_Shock.oblique_shock_relations(M0[i_sup],gamma[i_sup],theta*Units.deg,beta[i_sup])
        Pt1[i_sup]                                      = Ptr1[i_sup]*Pt0[i_sup]
        f_M1[i_sup]                                     = Isentropic.isentropic_relations(M1[i_sup], gamma[i_sup])[-1]
        
        Pt_out[i_sup] = Pt1[i_sup]
        
        f_M2[i_sup] = (1-Ks)*A0[i_sup]*f_M0[i_sup]*Pt0[i_sup]/(A2*Pt_out[i_sup]) #A1*f_M1[i_sup]/A2

        Mach[i_sup] = Isentropic.get_m(f_M2[i_sup], gamma[i_sup], 1)
        
        T_out = Isentropic.isentropic_relations(Mach, gamma)[0]*Tt_out  
        P_out = Isentropic.isentropic_relations(Mach, gamma)[1]*Pt_out  
        
        # -- Compute exit velocity and enthalpy
        h_out = Cp * T_out
        u_out = np.sqrt(2. * (ht_out - h_out))
        
        # pack computed quantities into outputs
        self.outputs.stagnation_temperature = Tt_out
        self.outputs.stagnation_pressure = Pt_out
        self.outputs.stagnation_enthalpy = ht_out
        self.outputs.mach_number = Mach
        self.outputs.static_temperature = T_out
        self.outputs.static_enthalpy = h_out
        self.outputs.velocity = u_out
        
        conditions.freestream.mach = Mach
        conditions.freestream.Pt_out = Pt_out
        conditions.freestream.Tout   = T_out
        
        m_dot0          = conditions.freestream.density * A0 * v0
        P_out           = Isentropic.isentropic_relations(Mach, gamma)[1]*Pt_out
        rho             = P_out/(R*T_out)

        conditions.mass_flow_rate = (1-Ks)*m_dot0

    ### Buggy drag calculations
        
    def compute_drag(self, conditions):
    
        '''
        Nomenclature/labeling of this section is inconsistent with the above
        but is consistent with Nikolai's methodology as presented in aircraft
        design
        '''
        
        # Unpack constants from freestream conditions
        gamma       = conditions.freestream.isentropic_expansion_factor
        R           = conditions.freestream.gas_specific_constant
        P_inf       = conditions.freestream.pressure
        M_inf       = np.atleast_2d(conditions.freestream.mach_number)
        rho_inf     = conditions.freestream.density
        Cp    = conditions.freestream.specific_heat_at_constant_pressure
    
        # unpack from inputs
        Tt_inf = self.inputs.stagnation_temperature
        Pt_inf = self.inputs.stagnation_pressure
        
        # compute relevant freestream quantities
        T_inf  = Isentropic.isentropic_relations(M_inf, gamma)[0] * Tt_inf
        v_inf  = np.sqrt(gamma*R*T_inf) * M_inf
        q_inf  = 1/2 * rho_inf * v_inf**2
        f_Minf = Isentropic.isentropic_relations(M_inf, gamma)[-1]
        
        Tt_out  = Tt_inf
        ht_out  = Cp*Tt_inf
        Pt_out  = np.ones_like(M_inf)
        Mach    = np.ones_like(M_inf)
        T_out   = np.ones_like(M_inf)
        f_M2    = np.ones_like(M_inf)        
    
        # unpack from self
        A2    = self.areas.engine_face # engine face area
        A_inf = conditions.freestream.area_initial_streamtube
        AC    = self.areas.inlet_capture
        A1    = self.areas.ramp_area  
        theta = self.angles.ramp_angle * Units.rad
        AS    = self.areas.drag_direct_projection
        Ks    = self.spillage_fraction
        Cp    = conditions.freestream.specific_heat_at_constant_pressure
        
        ht_out  = Cp*Tt_inf
        
        i_sup           = M_inf > 1.0
        i_sub           = M_inf <= 1.0
        
        
        # initialize values
        Pr1  = np.ones_like(Tt_inf)
        P1   = np.ones_like(Tt_inf)*P_inf
        Ptr1 = np.ones_like(Tt_inf) # stagnation pressure ratio after shock
        Pt1  = np.ones_like(Tt_inf)
        Tr1  = np.ones_like(Tt_inf)
        M1   = np.ones_like(Tt_inf)
        f_M1 = np.ones_like(Tt_inf)
        P1   = np.ones_like(Tt_inf)
        M1   = np.ones_like(Tt_inf)
        beta = np.ones_like(Tt_inf)
        
        Tt_out = Tt_inf
        
        D_add      = np.ones_like(Tt_inf)*0.0
        C_ps       = np.ones_like(Tt_inf)*0.0
        CD_add     = np.ones_like(Tt_inf)*0.0
        Ps_ov_Pinf = np.ones_like(Tt_inf)*0.0
        K_add      = np.ones_like(Tt_inf)*0.0
        
        # Subsonic flow into inlet (Assuming no normal shock at throat)
        # subsonic case
        f_M1[i_sub]      = (1-Ks)*(f_Minf[i_sub] * A_inf[i_sub])/A1
        M1[i_sub]        = Isentropic.get_m(f_M1[i_sub], gamma[i_sub], 1)
        P1[i_sub]        = Isentropic.isentropic_relations(M1[i_sub], gamma[i_sub])[1] * Pt_inf[i_sub]
        
        # supersonic case
        beta[i_sup]            = Oblique_Shock.theta_beta_mach(M_inf[i_sup],gamma[i_sup],theta,1)
        beta[i_sup]            = np.nan_to_num(beta[i_sup], nan=np.pi/2)
        
        # computing post shock quantities
        M1[i_sup], Pr1[i_sup], Tr1[i_sup], Ptr1[i_sup]  = Oblique_Shock.oblique_shock_relations(M_inf[i_sup],gamma[i_sup],theta*Units.deg,beta[i_sup])
        Pt1[i_sup]                                      = Ptr1[i_sup]*Pt_inf[i_sup]
        P1[i_sup]                                       = Isentropic.isentropic_relations(M1[i_sup], gamma[i_sup])[1]*Pt1[i_sup]
        f_M1[i_sup]                                     = Isentropic.isentropic_relations(M1[i_sup], gamma[i_sup])[-1]
        T1                                              = Isentropic.isentropic_relations(M1, gamma)[0]*Tt_inf
        h1                                              = Cp * T1
        v1                                              = np.sqrt(2. * (ht_out - h1))
        
        Pt_out[i_sup] = Pt1[i_sup]
        f_M2[i_sup] = (1-Ks)*A_inf[i_sup]*f_Minf[i_sup]*Pt_inf[i_sup]/(A2*Pt_out[i_sup]) #A1*f_M1[i_sup]/A2
        

        Mach[i_sup] = Isentropic.get_m(f_M2[i_sup], gamma[i_sup], 1)
        
        T_out = Isentropic.isentropic_relations(Mach, gamma)[0]*Tt_out  
        P_out = Isentropic.isentropic_relations(Mach, gamma)[1]*Pt_out  
        
        # -- Compute exit velocity and enthalpy
        h_out = Cp * T_out
        u_out = np.sqrt(2. * (ht_out - h_out))        
        
        mdot    = u_out*A2*P_out/(R*T_out)    
        for i, mach in enumerate(M_inf):
            
            if mach >= 0.7:
                            
                    # exposed area related drag
                    #Ps_ov_Pinf[i] = Ps[i]/P_inf[i]

                    ##Ps_ov_Pinf[i]    = P1[i]/P_inf[i] #Oblique_Shock.get_invisc_press_recov(theta/Units.deg, M_inf[i])
                    #C_ps[i]       = 2/(gamma[i]*M_inf[i]**2) * (Ps_ov_Pinf[i] - 1)
                    
                    #CD_add[i] = (P_inf[i]/q_inf[i]) * (A1/AC) * np.cos(theta) * ((P1[i]/P_inf[i])*(1+gamma[i]*M1[i]**2)-1) - 2*(A_inf[i]/AC) + C_ps[i]*(AS/AC)
                    
                    if mach >= 0.7 and mach <= 0.9:
                        
                        c1_fit_0709 = [-10.55390326, 15.71708277, -5.23617066]
                        c2_fit_0709 = [16.36281692, -24.54266271, 7.4994281]
                        c3_fit_0709 = [-4.86319239, 7.59775242, -1.85372994]  
                        
                        c1 = np.polyval(c1_fit_0709, M_inf[i])
                        c2 = np.polyval(c2_fit_0709, M_inf[i])
                        c3 = np.polyval(c3_fit_0709, M_inf[i])
                        
                        fit   = [c1, c2, c3]
                        K_add[i] = np.polyval(fit, A_inf[i]/AC)                             
                        
                    elif mach > 0.9 and mach <= 1.1:
                        
                        c1_fit_0911 = [2.64544806e-17, 3.60542191e-01]
                        c2_fit_0911 = [1.57079398e-16, -1.33508664e+00]
                        c3_fit_0911 = [-7.8265315e-16, 1.0450614e+00]   
                        
                        c1 = np.polyval(c1_fit_0911, M_inf[i])
                        c2 = np.polyval(c2_fit_0911, M_inf[i])
                        c3 = np.polyval(c3_fit_0911, M_inf[i])    
                        
                        fit   = [c1, c2, c3]
                        K_add[i] = np.polyval(fit, A_inf[i]/AC)                             
                        
                    elif mach > 1.1 and mach <= 1.4:
                        
                        c1_fit_else = [-102.15032982, 403.09453072, -527.81008066, 229.16933773]
                        c2_fit_else = [134.93205478, -539.18500576, 716.8828252, -317.08690229]
                        c3_fit_else = [-29.74762681, 122.74408883, -166.89910445, 75.70782011]
                        
                        c1 = np.polyval(c1_fit_else, M_inf[i])
                        c2 = np.polyval(c2_fit_else, M_inf[i])
                        c3 = np.polyval(c3_fit_else, M_inf[i])    
                        
                        fit   = [c1, c2, c3]
                        K_add[i] = np.polyval(fit, A_inf[i]/AC)                        
                        
                    elif mach > 1.4:

                        K_add[i] = 1
                        

                    #D_add[i]  = CD_add[i] * q_inf[i]* AC * K_add[i]

                    if K_add[i] > 1 or K_add[i] < 0.5:
                        K_add[i] = 1
                    #D_add[i] = K_add[i]*(-mdot[i]*v_inf[i] - A_inf[i]*(P1[i] - P_inf[i]) + mdot[i]*v1[i]*np.cos(theta) + A1*(P1[i] - P_inf[i])*np.cos(theta) + (P1[i]*np.sin(theta) - P_inf[i])*AS)
                    D_add[i] = K_add[i]*((mdot[i]*(-v_inf[i]+v1[i])+A1*(P1[i] - P_inf[i]) ))
                    
                    if i >= 1:
                        
                        if M_inf[i] < M_inf[i-1]:
                            D_add[i] = 0
                            D_add[i-1] = 0
                    if D_add[i] > 0:
                        D_add[i] = 0
                        

                      
        self.inlet_drag_calc = D_add
        conditions.freestream.inlet_drag = D_add
        
        mdot = A1*v1*P1/(R*T1)
        return D_add, mdot

    
    __call__ = compute
    