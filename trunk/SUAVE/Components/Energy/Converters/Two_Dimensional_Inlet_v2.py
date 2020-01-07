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
        self.areas.inlet_throat              = 0.0
        self.areas.inlet_capture             = 0.0
        self.areas.drag_direct_projection    = 0.0
        self.angles                          = Data()
        self.angles.ramp_angle               = 0.0 # should be in radians
        self.inputs.stagnation_temperature   = np.array([0.0])
        self.inputs.stagnation_pressure      = np.array([0.0])
        self.outputs.stagnation_temperature  = np.array([0.0])
        self.outputs.stagnation_pressure     = np.array([0.0])
        self.outputs.stagnation_enthalpy     = np.array([0.0])

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

        # unpack from inputs
        Tt0 = self.inputs.stagnation_temperature
        Pt0 = self.inputs.stagnation_pressure

        # unpack from self
        AE    = self.areas.engine_face # engine face area
        AC    = self.areas.inlet_capture # inlet capture area
        AT    = self.areas.inlet_throat # narrowest part of inlet (not used right now)
        theta = self.angles.ramp_angle/Units.deg # incoming angle for the shock converted to degrees
        Ks    = self.spillage_fraction; # fraction of incoming mass lost to spillage
        
        # Compute the mass flow rate into the engine
        T0              = Isentropic.isentropic_relations(M0, gamma)[0]*Tt0
        v0              = np.sqrt(gamma*R*T0)*M0
        m_dot0          = conditions.freestream.density * A0 * v0
        m_dotC          = (1-Ks) * m_dot0

        f_M0            = Isentropic.isentropic_relations(M0, gamma)[-1]
        
        i_sub           = M0 < 1.0
        i_sup           = M0 >= 1.0
        
        # initializing the arrays
        Tt_out  = Tt0
        ht_out  = Cp*Tt0
        Pt_out  = np.ones_like(M0)
        Mach    = np.ones_like(M0)
        T_out   = np.ones_like(M0)
        f_ME    = np.ones_like(M0)
        MC      = np.ones_like(M0)
        PrC    = np.ones_like(M0)
        TrC    = np.ones_like(M0)
        PtrC   = np.ones_like(M0)
        f_MC    = np.ones_like(M0)
        beta    = np.ones_like(M0)
        
        # Subsonic flow into inlet (Assuming no normal shock at throat)
        Pt_out[i_sub]   = Pt0[i_sub]
        f_ME[i_sub]     = (1-Ks)*(f_M0[i_sub] * A0[i_sub])/AE
        Mach[i_sub]     = Isentropic.get_m(f_ME[i_sub], gamma[i_sub], 1)
        T_out[i_sub]    = Isentropic.isentropic_relations(Mach[i_sub], gamma[i_sub])[0]*Tt_out[i_sub]        
        
        # Supersonic flow into inlet (Oblique shock on impact with ramp)
        beta[i_sup]                                     = Oblique_Shock.theta_beta_mach(M0[i_sup],gamma[i_sup],theta*Units.rad)
        MC[i_sup], PrC[i_sup], TrC[i_sup], PtrC[i_sup]  = Oblique_Shock.oblique_shock_relations(M0[i_sup],gamma[i_sup],theta*Units.rad,beta[i_sup])
        Pt_out[i_sup]                                   = PtrC[i_sup]*Pt0[i_sup]
        f_MC[i_sup]                                     = (1-Ks)*f_M0[i_sup]*A0[i_sup]*Pt0[i_sup]/(AC*Pt_out[i_sup]);
        f_ME[i_sup]                                     = f_MC[i_sup]*AC/AE;
        
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
        conditions.mass_flow_rate = m_dotC

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
    
        # unpack from inputs
        Tt_inf = self.inputs.stagnation_temperature
        Pt_inf = self.inputs.stagnation_pressure
        
        # compute relevant freestream quantities
        T_inf  = Isentropic.isentropic_relations(M_inf, gamma)[0] * Tt_inf
        v_inf  = np.sqrt(gamma*R*T_inf) * M_inf
        q_inf  = 1/2 * rho_inf * v_inf**2
        f_Minf = Isentropic.isentropic_relations(M_inf, gamma)[-1]
    
        # unpack from self
        A_inf = conditions.freestream.area_initial_streamtube
        AC    = self.areas.capture # engine face area
        A1    = self.areas.inlet_entrance # area of the inlet entrance
        theta = self.angles.ramp_angle * Units.rad
        AS    = self.areas.drag_direct_projection
        
        if all(M_inf >= 0.7):
            if all(A_inf > 0) and all(A_inf <= A1):
                
                f_Minf          = Isentropic.isentropic_relations(M_inf, gamma)[-1]
               
                f_MC_isentropic = (f_Minf * A_inf)/AC
                i_sub_shock     = np.logical_and(f_Minf <= 1.0, f_MC_isentropic > 1)
                i_sub_no_shock  = np.logical_and(f_Minf <= 1.0, f_MC_isentropic <= 1)
                i_sup           = f_Minf > 1.0
                
                
                # initialize values
                Pr_s = np.ones_like(Tt_inf)
                Ps   = np.ones_like(Tt_inf)*P_inf
                Pr_ts = np.ones_like(Tt_inf) # stagnation pressure ratio after shock
                P_ts = np.ones_like(Tt_inf)
                Ms   = np.ones_like(Tt_inf)
                f_M1 = np.ones_like(Tt_inf)
                Pr_1 = np.ones_like(Tt_inf)
                P1   = np.ones_like(Tt_inf)
                M1   = np.ones_like(Tt_inf)
                beta = np.ones_like(Tt_inf)
                
                # subsonic case
                f_M1[i_sub_no_shock]      = (f_Minf[i_sub_no_shock] * A_inf[i_sub_no_shock])/A1
                M1[i_sub_no_shock]        = Isentropic.get_m(f_M1[i_sub_no_shock], gamma[i_sub_no_shock], 1)
                P1[i_sub_no_shock]        = Isentropic.isentropic_relations(M1[i_sub_no_shock], gamma[i_sub_no_shock])[1] * Pt_inf[i_sub_no_shock]
                
                # subsonic with shock ->getting post shock quantities
                Ms[i_sub_shock], Pr_s[i_sub_shock] = Oblique_Shock.oblique_shock_relations(M_inf[i_sub_shock],gamma[i_sub_shock],0,90*np.pi/180.)[0:2]
                Pr_ts[i_sub_shock]                 = Oblique_Shock.oblique_shock_relations(M_inf[i_sub_shock],gamma[i_sub_shock],0,90*np.pi/180.)[3]
                Ps[i_sub_shock]                    = Pr_s[i_sub_shock]*P_inf[i_sub_shock]
                P_ts[i_sub_shock]                  = Pr_ts[i_sub_shock]*Pt_inf[i_sub_shock]
                
                f_M1[i_sub_shock] = 1/Pr_ts[i_sub_shock]*A_inf[i_sub_shock]/A1*f_Minf[i_sub_shock]
                M1[i_sub_shock]   = Isentropic.get_m(f_M1[i_sub_shock], gamma[i_sub_shock], 1)
                P1[i_sub_shock]   = Isentropic.isentropic_relations(M1[i_sub_shock],gamma[i_sub_shock])[1]*P_ts[i_sub_shock] 
                
                #M1[i_sub_shock], Pr_1[i_sub_shock] = Oblique_Shock.oblique_shock_relations(M_inf[i_sub_shock],gamma[i_sub_shock],0,90*np.pi/180.)[0:2]
                #P1[i_sub_shock]              = Pr_1[i_sub_shock]*P_inf[i_sub_shock]
                
                # supersonic case
                beta[i_sup]            = Oblique_Shock.theta_beta_mach(M_inf[i_sup],gamma[i_sup],theta)
                # computing post shock quantities
                Ms[i_sup], Pr_s[i_sup] = Oblique_Shock.oblique_shock_relations(M_inf[i_sup],gamma[i_sup],theta,beta[i_sup])[0:2]
                Pr_ts[i_sup]           = Oblique_Shock.oblique_shock_relations(M_inf[i_sup],gamma[i_sup],theta,beta[i_sup])[3]
                Ps[i_sup]              = Pr_s[i_sup]*P_inf[i_sup]
                
                f_M1[i_sup] = 1/Pr_ts[i_sup]*A_inf[i_sup]/A1*f_Minf[i_sup]
                M1[i_sup]   = Isentropic.get_m(f_M1[i_sup], gamma[i_sup], 1)
                P1[i_sup]   = Isentropic.isentropic_relations(M1[i_sup],gamma[i_sup])[1]*P_ts[i_sup] 
                
                #M1[i_sup], Pr_1[i_sup] = Oblique_Shock.oblique_shock_relations(M_inf[i_sup],gamma[i_sup],theta,beta[i_sup])[0:2]
                #P1[i_sup]              = Pr_1[i_sup]*P_inf[i_sup]
                
                # exposed area related drag
                Ps_ov_Pinf = Ps/P_inf
                #Ps_ov_Pinf = Oblique_Shock.get_invisc_press_recov(theta/Units.deg, M1)
                C_ps       = 2/(gamma*M_inf**2) * (Ps_ov_Pinf - 1)
                
                CD_add = (P_inf/q_inf) * (A1/AC) * np.cos(theta)*((P1/P_inf)*(1+gamma*M1**2)-1) - 2*(A_inf/AC) + C_ps*(AS/AC)
                
                i_14 = M_inf > 1.4
                i_0709 = np.logical_and(M_inf >= 0.7, M_inf <= 0.9)
                i_0911 = np.logical_and(M_inf > 0.9, M_inf <= 1.1)
                i_else = np.logical_and(M_inf > 1.1, M_inf <= 1.4)
                    
                c1_fit_0709 = [-10.55390326, 15.71708277, -5.23617066]
                c2_fit_0709 = [16.36281692, -24.54266271, 7.4994281]
                c3_fit_0709 = [-4.86319239, 7.59775242, -1.85372994]
                
                c1_fit_0911 = [2.64544806e-17, 3.60542191e-01]
                c2_fit_0911 = [1.57079398e-16, -1.33508664e+00]
                c3_fit_0911 = [-7.8265315e-16, 1.0450614e+00]
                
                c1_fit_else = [-102.15032982, 403.09453072, -527.81008066, 229.16933773]
                c2_fit_else = [134.93205478, -539.18500576, 716.8828252, -317.08690229]
                c3_fit_else = [-29.74762681, 122.74408883, -166.89910445, 75.70782011]
                
                c1   = np.ones_like(Tt_inf)
                c2   = np.ones_like(Tt_inf)
                c3   = np.ones_like(Tt_inf)
                            
                c1[i_0709] = np.polyval(c1_fit_0709, M_inf[i_0709])
                c2[i_0709] = np.polyval(c2_fit_0709, M_inf[i_0709])
                c3[i_0709] = np.polyval(c3_fit_0709, M_inf[i_0709])
                
                c1[i_0911] = np.polyval(c1_fit_0911, M_inf[i_0911])
                c2[i_0911] = np.polyval(c2_fit_0911, M_inf[i_0911])
                c3[i_0911] = np.polyval(c3_fit_0911, M_inf[i_0911])
                
                c1[i_else] = np.polyval(c1_fit_else, M_inf[i_else])
                c2[i_else] = np.polyval(c2_fit_else, M_inf[i_else])
                c3[i_else] = np.polyval(c3_fit_else, M_inf[i_else])
                
                # Use coefficients on theta_c to get the pressure recovery
                fit   = [c1, c2, c3]
                K_add = np.polyval(fit, A_inf/AC)
                
                K_add[i_14] = 1.
        
                D_add  = CD_add * q_inf* AC * K_add
            
            else:
                factor = 1
                ratio = A_inf/A1
                if any(A_inf < 0):
                    factor = abs(min(A_inf))
                else:
                    factor = (max(ratio)-1)
                D_add = factor*10**8
            
        else:
            D_add = np.ones_like(Tt_inf)*0.0
        
        print(D_add)
        return D_add

    __call__ = compute
    
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
            self.areas                           = Data()
            self.areas.capture                   = 0.0
            self.areas.throat                    = 0.0
            self.areas.inlet_entrance            = 0.0
            self.areas.drag_direct_projection    = 0.0
            self.angles                          = Data()
            self.angles.ramp_angle               = 0.0
            self.spillage_factor                 = 0.0
            self.inputs.stagnation_temperature   = np.array([0.0])
            self.inputs.stagnation_pressure      = np.array([0.0])
            self.outputs.stagnation_temperature  = np.array([0.0])
            self.outputs.stagnation_pressure     = np.array([0.0])
            self.outputs.stagnation_enthalpy     = np.array([0.0])
    
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
    
            # unpack from conditions
            gamma = conditions.freestream.isentropic_expansion_factor
            Cp = conditions.freestream.specific_heat_at_constant_pressure
            Po = conditions.freestream.pressure
            M0 = np.atleast_2d(conditions.freestream.mach_number)
            R = conditions.freestream.gas_specific_constant
    
            # unpack from inputs
            Tt_in = self.inputs.stagnation_temperature
            Pt_in = self.inputs.stagnation_pressure
    
            # unpack from self
            A0 = conditions.freestream.area_initial_streamtube
            AE = self.areas.capture # engine face area
            AC = self.areas.throat # narrowest part of inlet
            theta = self.angles.ramp_angle # incoming angle for the shock in degrees
            
            # Compute the mass flow rate into the engine
            T               = Isentropic.isentropic_relations(M0, gamma)[0]*Tt_in
            v               = np.sqrt(gamma*R*T)*M0
            mass_flow_rate  = conditions.freestream.density * A0 * v
            print("computed mass flow rate:", mass_flow_rate)
    
            f_M0            = Isentropic.isentropic_relations(M0, gamma)[-1]
            f_ME_isentropic = (f_M0 * A0)/AE
            
            f_MC_isentropic = (f_M0 * A0)/AC
            i_sub_shock     = np.logical_and(M0 <= 1.0, f_MC_isentropic > 1)
            i_sub_no_shock  = np.logical_and(M0 <= 1.0, f_MC_isentropic <= 1)
            i_sup           = M0 > 1.0
            
                
            # initializing the arrays
            Tt_out  = Tt_in
            ht_out  = Cp*Tt_in
            Pt_out  = np.ones_like(M0)
            Mach    = np.ones_like(M0)
            T_out   = np.ones_like(M0)
            f_ME    = np.ones_like(M0)
            MC      = np.ones_like(M0)
            Pr_c    = np.ones_like(M0)
            Tr_c    = np.ones_like(M0)
            Ptr_c   = np.ones_like(M0)
            f_MC    = np.ones_like(M0)
            beta    = np.ones_like(M0)
    
            # Conservation of mass properties to evaluate subsonic case
            Pt_out[i_sub_no_shock]   = Pt_in[i_sub_no_shock]
            f_ME[i_sub_no_shock]     = f_ME_isentropic[i_sub_no_shock]
            Mach[i_sub_no_shock]     = Isentropic.get_m(f_ME[i_sub_no_shock], gamma[i_sub_no_shock], 1)
            T_out[i_sub_no_shock]    = Isentropic.isentropic_relations(Mach[i_sub_no_shock], gamma[i_sub_no_shock])[0]*Tt_out[i_sub_no_shock]
            
            
            # Analysis of shocks for the subsonic with shock case case (normal shock at throat)
            MC[i_sub_shock], Pr_c[i_sub_shock], Tr_c[i_sub_shock], Ptr_c[i_sub_shock] = Oblique_Shock.oblique_shock_relations(M0[i_sub_shock],gamma[i_sub_shock],0,90*np.pi/180.)
            Pt_out[i_sub_shock] = Ptr_c[i_sub_shock]*Pt_in[i_sub_shock]
            f_MC[i_sub_shock] = Isentropic.isentropic_relations(MC[i_sub_shock], gamma[i_sub_shock])[-1]
            f_ME[i_sub_shock] = f_MC[i_sub_shock]*AC/AE
            
            Mach[i_sub_shock] = Isentropic.get_m(f_ME[i_sub_shock], gamma[i_sub_shock], 1)
            T_out[i_sub_shock] = Isentropic.isentropic_relations(Mach[i_sub_shock], gamma[i_sub_shock])[0]*Tt_out[i_sub_shock]
            
            # Analysis of shocks for the supersonic case
            beta[i_sup] = Oblique_Shock.theta_beta_mach(M0[i_sup],gamma[i_sup],theta*Units.rad)
            MC[i_sup], Pr_c[i_sup], Tr_c[i_sup], Ptr_c[i_sup] = Oblique_Shock.oblique_shock_relations(M0[i_sup],gamma[i_sup],theta*Units.rad,beta[i_sup])
            Pt_out[i_sup] = Ptr_c[i_sup]*Pt_in[i_sup]
            f_MC[i_sup] = Isentropic.isentropic_relations(MC[i_sup], gamma[i_sup])[-1]
            f_ME[i_sup] = f_MC[i_sup]*AC/AE
    
            Mach[i_sup] = Isentropic.get_m(f_ME[i_sup], gamma[i_sup], 1)
            T_out[i_sup] = Isentropic.isentropic_relations(Mach[i_sup], gamma[i_sup])[0]*Tt_out[i_sup]
            
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
            conditions.mass_flow_rate = mass_flow_rate
    
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
        
            # unpack from inputs
            Tt_inf = self.inputs.stagnation_temperature
            Pt_inf = self.inputs.stagnation_pressure
            
            # compute relevant freestream quantities
            T_inf  = Isentropic.isentropic_relations(M_inf, gamma)[0] * Tt_inf
            v_inf  = np.sqrt(gamma*R*T_inf) * M_inf
            q_inf  = 1/2 * rho_inf * v_inf**2
            f_Minf = Isentropic.isentropic_relations(M_inf, gamma)[-1]
        
            # unpack from self
            A_inf = conditions.freestream.area_initial_streamtube
            AC    = self.areas.capture # engine face area
            A1    = self.areas.inlet_entrance # area of the inlet entrance
            theta = self.angles.ramp_angle * Units.rad
            AS    = self.areas.drag_direct_projection
            
            if all(M_inf >= 0.7):
                if all(A_inf > 0) and all(A_inf <= A1):
                    
                    f_Minf          = Isentropic.isentropic_relations(M_inf, gamma)[-1]
                   
                    f_MC_isentropic = (f_Minf * A_inf)/AC
                    i_sub_shock     = np.logical_and(f_Minf <= 1.0, f_MC_isentropic > 1)
                    i_sub_no_shock  = np.logical_and(f_Minf <= 1.0, f_MC_isentropic <= 1)
                    i_sup           = f_Minf > 1.0
                    
                    
                    # initialize values
                    Pr_s = np.ones_like(Tt_inf)
                    Ps   = np.ones_like(Tt_inf)*P_inf
                    Pr_ts = np.ones_like(Tt_inf) # stagnation pressure ratio after shock
                    P_ts = np.ones_like(Tt_inf)
                    Ms   = np.ones_like(Tt_inf)
                    f_M1 = np.ones_like(Tt_inf)
                    Pr_1 = np.ones_like(Tt_inf)
                    P1   = np.ones_like(Tt_inf)
                    M1   = np.ones_like(Tt_inf)
                    beta = np.ones_like(Tt_inf)
                    
                    # subsonic case
                    f_M1[i_sub_no_shock]      = (f_Minf[i_sub_no_shock] * A_inf[i_sub_no_shock])/A1
                    M1[i_sub_no_shock]        = Isentropic.get_m(f_M1[i_sub_no_shock], gamma[i_sub_no_shock], 1)
                    P1[i_sub_no_shock]        = Isentropic.isentropic_relations(M1[i_sub_no_shock], gamma[i_sub_no_shock])[1] * Pt_inf[i_sub_no_shock]
                    
                    # subsonic with shock ->getting post shock quantities
                    Ms[i_sub_shock], Pr_s[i_sub_shock] = Oblique_Shock.oblique_shock_relations(M_inf[i_sub_shock],gamma[i_sub_shock],0,90*np.pi/180.)[0:2]
                    Pr_ts[i_sub_shock]                 = Oblique_Shock.oblique_shock_relations(M_inf[i_sub_shock],gamma[i_sub_shock],0,90*np.pi/180.)[3]
                    Ps[i_sub_shock]                    = Pr_s[i_sub_shock]*P_inf[i_sub_shock]
                    P_ts[i_sub_shock]                  = Pr_ts[i_sub_shock]*Pt_inf[i_sub_shock]
                    
                    f_M1[i_sub_shock] = 1/Pr_ts[i_sub_shock]*A_inf[i_sub_shock]/A1*f_Minf[i_sub_shock]
                    M1[i_sub_shock]   = Isentropic.get_m(f_M1[i_sub_shock], gamma[i_sub_shock], 1)
                    P1[i_sub_shock]   = Isentropic.isentropic_relations(M1[i_sub_shock],gamma[i_sub_shock])[1]*P_ts[i_sub_shock] 
                    
                    #M1[i_sub_shock], Pr_1[i_sub_shock] = Oblique_Shock.oblique_shock_relations(M_inf[i_sub_shock],gamma[i_sub_shock],0,90*np.pi/180.)[0:2]
                    #P1[i_sub_shock]              = Pr_1[i_sub_shock]*P_inf[i_sub_shock]
                    
                    # supersonic case
                    beta[i_sup]            = Oblique_Shock.theta_beta_mach(M_inf[i_sup],gamma[i_sup],theta)
                    # computing post shock quantities
                    Ms[i_sup], Pr_s[i_sup] = Oblique_Shock.oblique_shock_relations(M_inf[i_sup],gamma[i_sup],theta,beta[i_sup])[0:2]
                    Pr_ts[i_sup]           = Oblique_Shock.oblique_shock_relations(M_inf[i_sup],gamma[i_sup],theta,beta[i_sup])[3]
                    Ps[i_sup]              = Pr_s[i_sup]*P_inf[i_sup]
                    
                    f_M1[i_sup] = 1/Pr_ts[i_sup]*A_inf[i_sup]/A1*f_Minf[i_sup]
                    M1[i_sup]   = Isentropic.get_m(f_M1[i_sup], gamma[i_sup], 1)
                    P1[i_sup]   = Isentropic.isentropic_relations(M1[i_sup],gamma[i_sup])[1]*P_ts[i_sup] 
                    
                    #M1[i_sup], Pr_1[i_sup] = Oblique_Shock.oblique_shock_relations(M_inf[i_sup],gamma[i_sup],theta,beta[i_sup])[0:2]
                    #P1[i_sup]              = Pr_1[i_sup]*P_inf[i_sup]
                    
                    # exposed area related drag
                    Ps_ov_Pinf = Ps/P_inf
                    #Ps_ov_Pinf = Oblique_Shock.get_invisc_press_recov(theta/Units.deg, M1)
                    C_ps       = 2/(gamma*M_inf**2) * (Ps_ov_Pinf - 1)
                    
                    CD_add = (P_inf/q_inf) * (A1/AC) * np.cos(theta)*((P1/P_inf)*(1+gamma*M1**2)-1) - 2*(A_inf/AC) + C_ps*(AS/AC)
                    
                    i_14 = M_inf > 1.4
                    i_0709 = np.logical_and(M_inf >= 0.7, M_inf <= 0.9)
                    i_0911 = np.logical_and(M_inf > 0.9, M_inf <= 1.1)
                    i_else = np.logical_and(M_inf > 1.1, M_inf <= 1.4)
                        
                    c1_fit_0709 = [-10.55390326, 15.71708277, -5.23617066]
                    c2_fit_0709 = [16.36281692, -24.54266271, 7.4994281]
                    c3_fit_0709 = [-4.86319239, 7.59775242, -1.85372994]
                    
                    c1_fit_0911 = [2.64544806e-17, 3.60542191e-01]
                    c2_fit_0911 = [1.57079398e-16, -1.33508664e+00]
                    c3_fit_0911 = [-7.8265315e-16, 1.0450614e+00]
                    
                    c1_fit_else = [-102.15032982, 403.09453072, -527.81008066, 229.16933773]
                    c2_fit_else = [134.93205478, -539.18500576, 716.8828252, -317.08690229]
                    c3_fit_else = [-29.74762681, 122.74408883, -166.89910445, 75.70782011]
                    
                    c1   = np.ones_like(Tt_inf)
                    c2   = np.ones_like(Tt_inf)
                    c3   = np.ones_like(Tt_inf)
                                
                    c1[i_0709] = np.polyval(c1_fit_0709, M_inf[i_0709])
                    c2[i_0709] = np.polyval(c2_fit_0709, M_inf[i_0709])
                    c3[i_0709] = np.polyval(c3_fit_0709, M_inf[i_0709])
                    
                    c1[i_0911] = np.polyval(c1_fit_0911, M_inf[i_0911])
                    c2[i_0911] = np.polyval(c2_fit_0911, M_inf[i_0911])
                    c3[i_0911] = np.polyval(c3_fit_0911, M_inf[i_0911])
                    
                    c1[i_else] = np.polyval(c1_fit_else, M_inf[i_else])
                    c2[i_else] = np.polyval(c2_fit_else, M_inf[i_else])
                    c3[i_else] = np.polyval(c3_fit_else, M_inf[i_else])
                    
                    # Use coefficients on theta_c to get the pressure recovery
                    fit   = [c1, c2, c3]
                    K_add = np.polyval(fit, A_inf/AC)
                    
                    K_add[i_14] = 1.
            
                    D_add  = CD_add * q_inf* AC * K_add
                
                else:
                    factor = 1
                    ratio = A_inf/A1
                    if any(A_inf < 0):
                        factor = abs(min(A_inf))
                    else:
                        factor = (max(ratio)-1)
                    D_add = factor*10**8
                
            else:
                D_add = np.ones_like(Tt_inf)*0.0
            
            print(D_add)
            return D_add
    
        __call__ = compute