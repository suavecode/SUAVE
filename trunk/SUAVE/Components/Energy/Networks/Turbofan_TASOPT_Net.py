#Turbofan_Network.py
#
# Created:  Anil Variyar, Feb 2016
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp
import datetime
import time
from SUAVE.Core import Units


# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn
import copy


from SUAVE.Core import Data
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components.Propulsors.Propulsor import Propulsor


# ----------------------------------------------------------------------
#  Turbofan Network
# ----------------------------------------------------------------------

class Turbofan_TASOPT_Net(Propulsor):

    def __defaults__(self):

        #setting the default values
        self.tag = 'Turbo_Fan'
        self.number_of_engines = 1.0
        self.nacelle_diameter  = 1.0
        self.engine_length     = 1.0
        self.bypass_ratio      = 1.0
        self.areas             = Data()
        self.design_params     = None
        self.offdesign_params  = None
        self.max_iters         = 800 #1
        self.newton_relaxation = 1.0
        self.compressor_map_file = "Compressor_map.txt"
        self.cooling_flow = 1
        self.no_of_turbine_stages = 1


    _component_root_map = None


    def unpack(self):

        self.design_params    = Data()
        self.offdesign_params = Data()

        dp    = self.design_params

        # design parameters

        dp.aalpha  = self.bypass_ratio

        dp.pi_d    = self.inlet_nozzle.pressure_ratio
        dp.eta_d   = self.inlet_nozzle.polytropic_efficiency

        dp.pi_f    = self.fan.pressure_ratio
        dp.eta_f   = self.fan.polytropic_efficiency

        dp.pi_fn   = self.fan_nozzle.pressure_ratio
        dp.eta_fn  = self.fan_nozzle.polytropic_efficiency

        dp.pi_lc   = self.low_pressure_compressor.pressure_ratio
        dp.eta_lc  = self.low_pressure_compressor.polytropic_efficiency

        dp.pi_hc   = self.high_pressure_compressor.pressure_ratio
        dp.eta_hc  = self.high_pressure_compressor.polytropic_efficiency

        dp.Tt4     = self.combustor.turbine_inlet_temperature
        dp.pi_b    = self.combustor.pressure_ratio
        dp.eta_b   = self.combustor.efficiency
        dp.htf     = self.combustor.fuel_data.specific_energy

        dp.eta_lt  = self.low_pressure_turbine.polytropic_efficiency
        dp.etam_lt = self.low_pressure_turbine.mechanical_efficiency

        dp.eta_ht  = self.high_pressure_turbine.polytropic_efficiency
        dp.etam_ht = self.high_pressure_turbine.mechanical_efficiency

        dp.pi_tn   = self.core_nozzle.pressure_ratio
        dp.eta_tn  = self.core_nozzle.polytropic_efficiency


        dp.Pt5     = 1.0
        dp.M2      = 0.6
        dp.M2_5    = 0.6

        dp.HTR_f   = self.fan.hub_to_tip_ratio
        dp.HTR_hc  = self.high_pressure_compressor.hub_to_tip_ratio

        dp.Tref    = 288.15
        dp.Pref    = 101325.0

        dp.GF      = 1.0
        dp.Tt4_spec = np.copy(dp.Tt4)
        dp.N_spec   = 1.0
        dp.Tt4_Tt2  = 1.0

        # offdesign parameters

        odp = copy.deepcopy(dp)
        self.offdesign_params = odp



    def evaluate(self,conditions,design_run,ep_eval = None):

        if ep_eval:
            ep = ep_eval
        elif design_run == True:
            ep = self.design_params
        else:
            ep = self.offdesign_params

        # temporary adjustment for testing
        try:
            ep.aalpha  = ep.aalpha.T

            ep.pi_d    = ep.pi_d.T
            ep.eta_d   = ep.eta_d.T

            ep.pi_f    = ep.pi_f.T
            ep.eta_f   = ep.eta_f.T

            ep.pi_fn   = ep.pi_fn.T
            ep.eta_fn  = ep.eta_fn.T

            ep.pi_lc   = ep.pi_lc.T
            ep.eta_lc  = ep.eta_lc.T

            ep.pi_hc   = ep.pi_hc.T
            ep.eta_hc  = ep.eta_hc.T

            ep.Tt4     = ep.Tt4.T
            ep.pi_b    = ep.pi_b.T
            ep.eta_b   = ep.eta_b.T

            ep.eta_lt  = ep.eta_lt.T
            ep.etam_lt = ep.etam_lt.T

            ep.eta_ht  = ep.eta_ht.T
            ep.etam_ht = ep.etam_ht.T

            ep.pi_tn   = ep.pi_tn.T
            ep.eta_tn  = ep.eta_tn.T

            ep.mhc = ep.mhc.T
            ep.mlc = ep.mlc.T
            ep.mf  = ep.mf.T

            ep.Pt5  = ep.Pt5.T
            ep.Tt4_Tt2 = ep.Tt4_Tt2.T
        except AttributeError:
            pass

        P0 = conditions.freestream.pressure
        T0 = conditions.freestream.temperature
        M0 = conditions.freestream.mach_number
        a0 = conditions.freestream.speed_of_sound
        u0 = conditions.freestream.velocity
        gamma = conditions.freestream.gamma
        Cp    = conditions.freestream.specific_heat
        R     = conditions.freestream.gas_specific_constant
        g     = conditions.freestream.gravity
        throttle = conditions.propulsion.throttle
        N_spec  = throttle
        ep.N_spec = throttle
        results = Data()

        # Ram calculations
        Dh = .5*u0*u0
        h0  = Cp*T0

        ram = self.ram
        ram.inputs.total_temperature = T0
        ram.inputs.total_pressure    = P0
        ram.inputs.total_enthalpy    = h0
        ram.inputs.delta_enthalpy    = Dh
        ram.inputs.working_fluid.specific_heat     = Cp
        ram.inputs.working_fluid.gamma             = gamma
        ram.compute_flow()

        Tt0 = ram.outputs.total_temperature
        Pt0 = ram.outputs.total_pressure
        ht0 = ram.outputs.total_enthalpy


        # Inlet Nozzle (stages 1.8 and 1.9 are assumed to match)

        inlet_nozzle = self.inlet_nozzle
        inlet_nozzle.inputs.total_temperature = Tt0
        inlet_nozzle.inputs.total_pressure    = Pt0
        inlet_nozzle.inputs.total_enthalpy    = ht0
        inlet_nozzle.compute_flow()

        Tt2 = inlet_nozzle.outputs.total_temperature
        Pt2 = inlet_nozzle.outputs.total_pressure
        ht2 = inlet_nozzle.outputs.total_enthalpy

        Tt1_9 = Tt2 # These are needed for later calculations
        Pt1_9 = Pt2
        ht1_9 = ht2


        # Fan

        fan = self.fan

        if design_run:
            fan.set_design_condition()
        else:
            fan.corrected_mass_flow   = ep.mf
            fan.pressure_ratio        = ep.pi_f
            fan.compute_performance()
            ep.eta_f = fan.polytropic_efficiency
            Nfn      = fan.corrected_speed
            dNfn_pif = fan.speed_change_by_pressure_ratio
            dNfn_mf  = fan.speed_change_by_mass_flow

        fan.inputs.working_fluid.specific_heat = Cp
        fan.inputs.working_fluid.gamma         = gamma
        fan.inputs.working_fluid.R             = R
        fan.inputs.total_temperature           = Tt2
        fan.inputs.total_pressure              = Pt2
        fan.inputs.total_enthalpy              = ht2
        fan.compute()

        Tt2_1 = fan.outputs.total_temperature
        Pt2_1 = fan.outputs.total_pressure
        ht2_1 = fan.outputs.total_enthalpy


        # Fan Nozzle

        fan_nozzle = self.fan_nozzle
        fan_nozzle.inputs.working_fluid.specific_heat = Cp
        fan_nozzle.inputs.working_fluid.gamma         = gamma
        fan_nozzle.inputs.working_fluid.R             = R
        fan_nozzle.inputs.total_temperature = Tt2_1
        fan_nozzle.inputs.total_pressure    = Pt2_1
        fan_nozzle.inputs.total_enthalpy    = ht2_1

        fan_nozzle.compute()

        Tt7 = fan_nozzle.outputs.total_temperature
        Pt7 = fan_nozzle.outputs.total_pressure
        ht7 = fan_nozzle.outputs.total_enthalpy


        # Low Pressure Compressor

        lpc = self.low_pressure_compressor
        if design_run:
            lpc.set_design_condition()
        else:
            lpc.corrected_mass_flow   = ep.mlc
            lpc.pressure_ratio        = ep.pi_lc
            lpc.compute_performance()
            ep.eta_lc = lpc.polytropic_efficiency
            Nln       = lpc.corrected_speed
            dNln_pilc = lpc.speed_change_by_pressure_ratio
            dNln_mlc  = lpc.speed_change_by_mass_flow

        lpc.inputs.working_fluid.specific_heat = Cp
        lpc.inputs.working_fluid.gamma         = gamma
        lpc.inputs.working_fluid.R             = R
        lpc.inputs.total_temperature           = Tt2
        lpc.inputs.total_pressure              = Pt2
        lpc.inputs.total_enthalpy              = ht2
        lpc.compute()

        Tt2_5 = lpc.outputs.total_temperature
        Pt2_5 = lpc.outputs.total_pressure
        ht2_5 = lpc.outputs.total_enthalpy


        # High Pressure Compressor

        hpc = self.high_pressure_compressor
        if design_run:
            hpc.set_design_condition()
        else:
            hpc.corrected_mass_flow   = ep.mhc
            hpc.pressure_ratio        = ep.pi_hc
            hpc.compute_performance()
            ep.eta_hc = hpc.polytropic_efficiency
            Nhn       = hpc.corrected_speed
            dNhn_pihc = hpc.speed_change_by_pressure_ratio
            dNhn_mhc  = hpc.speed_change_by_mass_flow

        hpc.inputs.working_fluid.specific_heat = Cp
        hpc.inputs.working_fluid.gamma         = gamma
        hpc.inputs.working_fluid.R             = R
        hpc.inputs.total_temperature           = Tt2_5
        hpc.inputs.total_pressure              = Pt2_5
        hpc.inputs.total_enthalpy              = ht2_5
        hpc.compute()

        Tt3 = hpc.outputs.total_temperature
        Pt3 = hpc.outputs.total_pressure
        ht3 = hpc.outputs.total_enthalpy


        # Combustor

        # Some inputs are only used if a cooling combustor is used
        combustor = self.combustor
        combustor.inputs.working_fluid.specific_heat = Cp
        combustor.inputs.working_fluid.gamma         = gamma
        combustor.inputs.working_fluid.R             = R
        combustor.inputs.total_temperature           = Tt3
        combustor.inputs.total_pressure              = Pt3
        combustor.inputs.total_enthalpy              = ht3

        combustor.turbine_inlet_temperature = ep.Tt4

        combustor.compute()

        Tt4 = combustor.outputs.total_temperature
        Pt4 = combustor.outputs.total_pressure
        ht4 = combustor.outputs.total_enthalpy
        f   = combustor.outputs.normalized_fuel_flow

        Tt4_1 = Tt4
        Pt4_1 = Pt4
        ht4_1 = ht4


        # Update the bypass ratio

        if design_run == False:
            mcore = ep.mlc*(Pt2/ep.Pref)/np.sqrt(Tt2/ep.Tref)/(1.0 + f)
            mfan  = ep.mf*(Pt2/ep.Pref)/np.sqrt(Tt2/ep.Tref)/(1.0 + f)
            ep.aalpha = mfan/mcore


        # High Pressure Turbine

        deltah_ht = -1./(1.+f)*1./ep.etam_ht*(ht3-ht2_5)

        hpt = self.high_pressure_turbine

        hpt.inputs.working_fluid.specific_heat = Cp
        hpt.inputs.working_fluid.gamma         = gamma
        hpt.inputs.total_temperature           = Tt4_1
        hpt.inputs.total_pressure              = Pt4_1
        hpt.inputs.total_enthalpy              = ht4_1
        hpt.inputs.delta_enthalpy              = deltah_ht

        hpt.compute()

        Tt4_5 = hpt.outputs.total_temperature
        Pt4_5 = hpt.outputs.total_pressure
        ht4_5 = hpt.outputs.total_enthalpy


        # Low Pressure Turbine

        if design_run == True:

            deltah_lt =  -1./(1.+f)*1./ep.etam_lt*((ht2_5 - ht1_9)+ ep.aalpha*(ht2_1 - ht2))

            lpt = self.low_pressure_turbine

            lpt.inputs.working_fluid.specific_heat = Cp
            lpt.inputs.working_fluid.gamma         = gamma
            lpt.inputs.total_temperature           = Tt4_5
            lpt.inputs.total_pressure              = Pt4_5
            lpt.inputs.total_enthalpy              = ht4_5
            lpt.inputs.delta_enthalpy              = deltah_lt

            lpt.compute()

            Tt4_9 = lpt.outputs.total_temperature
            Pt4_9 = lpt.outputs.total_pressure
            ht4_9 = lpt.outputs.total_enthalpy

        else:

            # Low pressure turbine off design case
            # A different setup is used for convergence per the TASOPT manual

            pi_lt = 1.0/ep.pi_tn*(ep.Pt5/Pt4_5)
            Pt4_9 = Pt4_5*pi_lt
            Tt4_9 = Tt4_5*pi_lt**((gamma-1.)*ep.eta_lt/(gamma))
            ht4_9 = Cp*Tt4_9


        # Core Nozzle
        # Sometimes tn is used for turbine nozzle

        core_nozzle = self.core_nozzle
        core_nozzle.inputs.working_fluid.specific_heat = Cp
        core_nozzle.inputs.working_fluid.gamma         = gamma
        core_nozzle.inputs.working_fluid.R             = R
        core_nozzle.inputs.total_temperature = Tt4_9
        core_nozzle.inputs.total_pressure    = Pt4_9
        core_nozzle.inputs.total_enthalpy    = ht4_9

        core_nozzle.compute()

        Tt5 = core_nozzle.outputs.total_temperature
        Pt5 = core_nozzle.outputs.total_pressure
        ht5 = core_nozzle.outputs.total_enthalpy


        # Core Exhaust

        # set pressure ratio to atmospheric

        core_exhaust = self.core_exhaust
        core_exhaust.pressure_ratio = P0/Pt5
        core_exhaust.inputs.working_fluid.specific_heat = Cp
        core_exhaust.inputs.working_fluid.gamma         = gamma
        core_exhaust.inputs.working_fluid.R             = R
        core_exhaust.inputs.total_temperature = Tt5
        core_exhaust.inputs.total_pressure    = Pt5
        core_exhaust.inputs.total_enthalpy    = ht5

        core_exhaust.compute()

        T6 = core_exhaust.outputs.static_temperature
        u6 = core_exhaust.outputs.flow_speed


        # Fan Exhaust

        fan_exhaust = self.fan_exhaust
        fan_exhaust.pressure_ratio = P0/Pt7
        fan_exhaust.inputs.working_fluid.specific_heat = Cp
        fan_exhaust.inputs.working_fluid.gamma         = gamma
        fan_exhaust.inputs.working_fluid.R             = R
        fan_exhaust.inputs.total_temperature = Tt7
        fan_exhaust.inputs.total_pressure    = Pt7
        fan_exhaust.inputs.total_enthalpy    = ht7

        fan_exhaust.compute()

        T8 = fan_exhaust.outputs.static_temperature
        u8 = fan_exhaust.outputs.flow_speed


        # Calculate Specific Thrust

        thrust = self.thrust
        thrust.inputs.normalized_fuel_flow_rate = f
        thrust.inputs.core_exhaust_flow_speed   = u6
        thrust.inputs.fan_exhaust_flow_speed    = u8
        thrust.inputs.bypass_ratio              = ep.aalpha

        conditions.freestream.speed_of_sound = a0
        conditions.freestream.velocity       = u0
        thrust.compute(conditions)

        Fsp = thrust.outputs.specific_thrust
        Isp = thrust.outputs.specific_impulse
        sfc = thrust.outputs.specific_fuel_consumption

        if design_run == True:

            # Determine design thrust per engine
            FD = self.thrust.total_design/(self.number_of_engines)

            # Core Mass Flow Calculation
            mdot_core = FD/(Fsp*a0*(1.+ep.aalpha))


            # Fan Sizing

            fan.size(mdot_core,ep.M2,ep.aalpha,ep.HTR_f)
            A2 = fan.entrance_area


            # High Pressure Compressor Sizing

            hpc.size(mdot_core,ep.M2,ep.aalpha,ep.HTR_hc)
            A2_5 = hpc.entrance_area


            # Fan Nozzle Area

            fan_nozzle.size(mdot_core,u8,T8,P0,ep.aalpha)
            A7 = fan_nozzle.exit_area


            # Core Nozzle Area

            core_nozzle.size(mdot_core,u6,T6,P0)
            A5 = core_nozzle.exit_area


            # spool speed

            NlcD = 1.0
            NhcD = 1.0

            # non dimensionalization

            NlD = NlcD*1.0/np.sqrt(Tt1_9/ep.Tref)
            NhD = NhcD*1.0/np.sqrt(Tt2_5/ep.Tref)

            mhtD = (1.0+f)*mdot_core*np.sqrt(Tt4_1/ep.Tref)/(Pt4_1/ep.Pref)
            mltD = (1.0+f)*mdot_core*np.sqrt(Tt4_5/ep.Tref)/(Pt4_5/ep.Pref)

            mhcD = (1.0+f)*mdot_core*np.sqrt(Tt2_5/ep.Tref)/(Pt2_5/ep.Pref)
            mlcD = (1.0+f)*mdot_core*np.sqrt(Tt2/ep.Tref)/(Pt2/ep.Pref)
            mfD  = (1.0+f)*ep.aalpha*mdot_core*np.sqrt(Tt2/ep.Tref)/(Pt2/ep.Pref)

            # Update engine parameters

            ep = self.design_params

            ep.A5   = A5
            ep.A7   = A7
            ep.A2   = A2
            ep.A2_5 = A2_5


            ep.mhtD = mhtD
            ep.mltD = mltD

            ep.mhcD = mhcD
            ep.mlcD = mlcD
            ep.mfD  = mfD
            fan.speed_map.design_mass_flow      = mfD
            fan.efficiency_map.design_mass_flow = mfD
            lpc.speed_map.design_mass_flow      = mlcD
            lpc.efficiency_map.design_mass_flow = mlcD
            hpc.speed_map.design_mass_flow      = mhcD
            hpc.efficiency_map.design_mass_flow = mhcD


            ep.NlcD = NlcD
            ep.NhcD = NhcD
            ep.NlD  = NlD
            ep.NhD  = NhD


            ep.mhtD = mhtD
            ep.mltD = mltD

            ep.mhc = mhcD
            ep.mlc = mlcD
            ep.mf  = mfD

            ep.Nl = NlcD
            ep.Nh = NhcD
            ep.Nf  = 1.0

            ep.Nln  = NlD
            ep.Nhn  = NhD
            ep.Nfn  = NlD
            fan.speed_map.Nd   = NlD
            lpc.speed_map.Nd   = NlD
            hpc.speed_map.Nd   = NhD

            ep.Pt5  = Pt5
            ep.Tt4_Tt2 = ep.Tt4/Tt2




        else:

            ep.Tt3 = Tt3
            ep.Pt3 = Pt3
            ##lpc
            ep.Nl = np.sqrt(Tt2/ep.Tref)*Nln

            ##hpc
            ep.Nh = np.sqrt(Tt2_5/ep.Tref)*Nhn

            ##fpc
            ep.Nf = np.sqrt(Tt1_9/ep.Tref)*Nfn


            ep.Tt4_spec = ep.Tt4_Tt2*Tt2*(throttle)


            mdot_core = ep.mlc*Pt2/ep.Pref/np.sqrt(Tt2/ep.Tref)

            thrust.inputs.normalized_fuel_flow_rate = f
            thrust.inputs.core_exhaust_flow_speed   = u6
            thrust.inputs.fan_exhaust_flow_speed    = u8
            thrust.inputs.bypass_ratio              = ep.aalpha

            conditions.freestream.speed_of_sound = a0
            conditions.freestream.velocity       = u0
            thrust.compute(conditions)

            Fsp = thrust.outputs.specific_thrust
            Isp = thrust.outputs.specific_impulse
            sfc = thrust.outputs.specific_fuel_consumption

            F    = Fsp*(1+ep.aalpha)*mdot_core*a0
            mdot = mdot_core*f

            ep.mdot_core = mdot_core


            # Fan nozzle flow properties

            fan_nozzle.compute_static(u8,T8,P0)
            u7   = fan_nozzle.outputs.flow_speed
            rho7 = fan_nozzle.outputs.static_density


            # Core nozzle flow properties

            core_nozzle.compute_static(u6,T6,P0)
            u5   = core_nozzle.outputs.flow_speed
            rho5 = core_nozzle.outputs.static_density




            #compute offdesign residuals

            # Low pressure turbine off design case
            # A different setup is used for convergence per the TASOPT manual

            deltah_lt =  -1./(1.+f)*1./ep.etam_lt*((ht2_5 - ht1_9)+ ep.aalpha*(ht2_1 - ht2))
            Tt4_9     = Tt4_5 + deltah_lt/Cp
            Pt4_9     = Pt4_5*(Tt4_9/Tt4_5)**(gamma/((gamma-1.)*ep.eta_lt))
            ht4_9     = ht4_5 + deltah_lt

            # temporary adjustment for testing
            try:
                ep.aalpha  = ep.aalpha.T

                ep.pi_d    = ep.pi_d.T
                ep.eta_d   = ep.eta_d.T

                ep.pi_f    = ep.pi_f.T
                ep.eta_f   = ep.eta_f.T

                ep.pi_fn   = ep.pi_fn.T
                ep.eta_fn  = ep.eta_fn.T

                ep.pi_lc   = ep.pi_lc.T
                ep.eta_lc  = ep.eta_lc.T

                ep.pi_hc   = ep.pi_hc.T
                ep.eta_hc  = ep.eta_hc.T

                ep.Tt4     = ep.Tt4.T
                ep.pi_b    = ep.pi_b.T
                ep.eta_b   = ep.eta_b.T

                ep.eta_lt  = ep.eta_lt.T
                ep.etam_lt = ep.etam_lt.T

                ep.eta_ht  = ep.eta_ht.T
                ep.etam_ht = ep.etam_ht.T

                ep.pi_tn   = ep.pi_tn.T
                ep.eta_tn  = ep.eta_tn.T

                ep.mhc = ep.mhc.T
                ep.mlc = ep.mlc.T
                ep.mf  = ep.mf.T

                ep.Pt5  = ep.Pt5.T
                ep.Tt4_Tt2 = ep.Tt4_Tt2.T
            except AttributeError:
                pass


            Res = np.zeros([8,M0.shape[0]])


            # temporary transposes
            Res[0,:] = (ep.Nf*ep.GF - ep.Nl).T
            Res[1,:] = ((1.+f)*ep.mhc.T*np.sqrt(Tt4_1/Tt2_5)*Pt2_5/Pt4_1 - ep.mhtD).T
            Res[2,:] = ((1.+f)*ep.mhc.T*np.sqrt(Tt4_5/Tt2_5)*Pt2_5/Pt4_5 - ep.mltD).T
            Res[3,:] = (ep.mf.T*np.sqrt(ep.Tref/Tt2)*Pt2/ep.Pref - rho7*u7*ep.A7).T
            Res[4,:] = ((1.+f)*ep.mhc.T*np.sqrt(ep.Tref/Tt2_5)*Pt2_5/ep.Pref - rho5*u5*ep.A5).T
            Res[5,:] = (ep.mlc.T*np.sqrt(ep.Tref/Tt1_9)*Pt1_9/ep.Pref - ep.mhc.T*np.sqrt(ep.Tref/Tt2_5)*Pt2_5/ep.Pref).T
            Res[6,:] = (ep.Nl - ep.N_spec).T


            Res[7,:] = (ep.Pt5.T - Pt4_9*ep.pi_tn.T).T

            results.Res = Res
            results.F   = F
            results.mdot = mdot


        results.Fsp  = Fsp
        results.Isp  = Isp
        results.sfc  = sfc



        return results



    def size(self,mach_number,altitude,delta_isa = 0.):

        #Unpack components
        atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmo_data = atmosphere.compute_values(altitude,delta_isa)

        p   = atmo_data.pressure
        T   = atmo_data.temperature
        rho = atmo_data.density
        a   = atmo_data.speed_of_sound
        mu  = atmo_data.dynamic_viscosity

        # setup conditions
        conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
        freestream_gas = SUAVE.Attributes.Gases.Air()

        # Note that these calculations are not accounting for temperature, and
        # therefore vary slightly from those used to calculate atmosphere parameters.
        # This means values such as speed of sound will vary slightly if computed with
        # these outputs directly.
        gamma = freestream_gas.compute_gamma()
        Cp    = freestream_gas.compute_cp()
        R     = freestream_gas.gas_specific_constant


        # freestream conditions

        conditions.freestream.altitude           = np.atleast_1d(altitude)
        conditions.freestream.mach_number        = np.atleast_1d(mach_number)

        conditions.freestream.pressure           = np.atleast_1d(p)
        conditions.freestream.temperature        = np.atleast_1d(T)
        conditions.freestream.density            = np.atleast_1d(rho)
        conditions.freestream.dynamic_viscosity  = np.atleast_1d(mu)
        conditions.freestream.gravity            = np.atleast_1d(9.81)
        conditions.freestream.gamma              = np.atleast_2d(gamma)
        conditions.freestream.specific_heat      = np.atleast_2d(Cp)
        conditions.freestream.gas_specific_constant = np.atleast_2d(R)
        conditions.freestream.speed_of_sound     = np.atleast_1d(a)
        conditions.freestream.velocity           = conditions.freestream.mach_number * conditions.freestream.speed_of_sound

        # propulsion conditions
        conditions.propulsion.throttle           =  np.atleast_1d(1.0)

        design_run = True
        results = self.evaluate(conditions, design_run)

        self.offdesign_params = deepcopy(self.design_params)




        ones_1col = np.ones([1,1])
        altitude      = ones_1col*0.0
        mach_number   = ones_1col*0.0
        throttle      = ones_1col*1.0

        #call the atmospheric model to get the conditions at the specified altitude
        atmosphere_sls = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmo_data = atmosphere_sls.compute_values(altitude,0.0)

        p   = atmo_data.pressure
        T   = atmo_data.temperature
        rho = atmo_data.density
        a   = atmo_data.speed_of_sound
        mu  = atmo_data.dynamic_viscosity

        # setup conditions
        conditions_sls = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
        freestream_gas = SUAVE.Attributes.Gases.Air()

        # Note that these calculations are not accounting for temperature, and
        # therefore vary slightly from those used to calculate atmosphere parameters.
        # This means values such as speed of sound will vary slightly if computed with
        # these outputs directly.
        gamma = freestream_gas.compute_gamma()
        Cp    = freestream_gas.compute_cp()
        R     = freestream_gas.gas_specific_constant


        # freestream conditions

        conditions_sls.freestream.altitude           = np.atleast_1d(altitude)
        conditions_sls.freestream.mach_number        = np.atleast_1d(mach_number)

        conditions_sls.freestream.pressure           = np.atleast_1d(p)
        conditions_sls.freestream.temperature        = np.atleast_1d(T)
        conditions_sls.freestream.density            = np.atleast_1d(rho)
        conditions_sls.freestream.dynamic_viscosity  = np.atleast_1d(mu)
        conditions_sls.freestream.gravity            = np.atleast_1d(9.81)
        conditions_sls.freestream.gamma              = np.atleast_2d(gamma)
        conditions_sls.freestream.specific_heat      = np.atleast_2d(Cp)
        conditions_sls.freestream.gas_specific_constant = np.atleast_2d(R)
        conditions_sls.freestream.speed_of_sound     = np.atleast_1d(a)
        conditions_sls.freestream.velocity           = conditions_sls.freestream.mach_number * conditions_sls.freestream.speed_of_sound

        # propulsion conditions
        conditions_sls.propulsion.throttle           =  np.atleast_1d(throttle)

        state_sls = Data()
        state_sls.numerics = Data()
        state_sls.conditions = conditions_sls
        results_sls = self.offdesign(state_sls)



        self.sealevel_static_thrust = results_sls.F

        return results






    def evaluate_thrust(self,state):

        #imports
        conditions = state.conditions
        numerics   = state.numerics
        throttle = conditions.propulsion.throttle

        local_state = copy.deepcopy(state)
        local_throttle = copy.deepcopy(throttle)

        local_throttle[throttle<0.6] = 0.6
        local_throttle[throttle>1.0] = 1.0

        local_state.conditions.propulsion.throttle = local_throttle


        #freestream properties
        T0 = conditions.freestream.temperature
        p0 = conditions.freestream.pressure
        M0 = conditions.freestream.mach_number


        F = np.zeros([len(T0),3])
        mdot0 = np.zeros([len(T0),1])

        results_eval = self.offdesign(local_state)

        local_scale = throttle/local_throttle


        F[:,0] = (results_eval.F*local_scale)[:,0]
        mdot0[:,0] = (results_eval.mdot*local_scale)[:,0]
        S = results_eval.TSFC


        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot0
        results.sfc                 = S


        return results




    def offdesign(self,state):

        # Start engine parameters as the design parameters
        ep = deepcopy(self.design_params)
        self.offdesign_params = ep

        conditions = state.conditions
        throttle = conditions.propulsion.throttle

        segment_length = len(throttle)

        self.init_vectors(conditions,ep)

        design_run = False
        results = self.evaluate(conditions,design_run)
        R = results.Res
        bp = self.set_baseline_params(len(throttle))


        d_bp = np.zeros([8,segment_length])


        if(np.linalg.norm(R)>1e-8):



            for iiter in range(0,self.max_iters):


                J = self.jacobian(conditions,bp,R)

                for iarr in range(0,segment_length):
                    d_bp[:,iarr] = -np.linalg.solve(J[:,:,iarr], R[:,iarr])



                bp = bp + self.newton_relaxation*d_bp
                self.update_engine_params(bp)
                design_run = False
                results = self.evaluate(conditions,design_run)
                R = results.Res


                if(np.linalg.norm(R)<1e-6):
                    break

                if iiter == (self.max_iters-1):
                    aaa = 0






        results_offdesign = Data()
        results_offdesign.F    = results.F*self.number_of_engines
        results_offdesign.TSFC = results.sfc
        results_offdesign.mdot = results.mdot*self.number_of_engines
        results_offdesign.Tt4  = ep.Tt4_spec
        results_offdesign.Tt3  = ep.Tt3
        results_offdesign.Pt3  = ep.Pt3
        results_offdesign.pi_lc = ep.pi_lc
        results_offdesign.pi_hc = ep.pi_hc
        results_offdesign.pi_f = ep.pi_f
        results_offdesign.flow = ep.mdot_core
        results_offdesign.aalpha = ep.aalpha
        results_offdesign.mdot_fan = ep.aalpha*ep.mdot_core
        results_offdesign.Nl = self.offdesign_params.Nl
        results_offdesign.Nf = self.offdesign_params.Nf
        results_offdesign.Nh = self.offdesign_params.Nh
        results_offdesign.mlc = self.offdesign_params.mlc
        results_offdesign.mhc = self.offdesign_params.mhc
        results_offdesign.mf = self.offdesign_params.mf

        #print results_offdesign.F,throttle.T,results_offdesign.TSFC

        return results_offdesign








    def init_vectors(self,conditions,ep):

        segment_length = len(conditions.propulsion.throttle)
        onesv = np.ones([1,segment_length])

        ep.aalpha  = ep.aalpha*onesv

        ep.pi_d    = ep.pi_d*onesv
        ep.eta_d   = ep.eta_d*onesv

        ep.pi_f    = ep.pi_f*onesv
        ep.eta_f   = ep.eta_f*onesv

        ep.pi_fn   = ep.pi_fn*onesv
        ep.eta_fn  = ep.eta_fn*onesv

        ep.pi_lc   = ep.pi_lc*onesv
        ep.eta_lc  = ep.eta_lc*onesv

        ep.pi_hc   = ep.pi_hc*onesv
        ep.eta_hc  = ep.eta_hc*onesv

        ep.Tt4     = ep.Tt4*onesv
        ep.pi_b    = ep.pi_b*onesv
        ep.eta_b   = ep.eta_b*onesv

        ep.eta_lt  = ep.eta_lt*onesv
        ep.etam_lt = ep.etam_lt*onesv

        ep.eta_ht  = ep.eta_ht*onesv
        ep.etam_ht = ep.etam_ht*onesv

        ep.pi_tn   = ep.pi_tn*onesv
        ep.eta_tn  = ep.eta_tn*onesv

        ep.mhc = ep.mhc*onesv
        ep.mlc = ep.mlc*onesv
        ep.mf  = ep.mf*onesv

        ep.Pt5  = ep.Pt5*onesv
        ep.Tt4_Tt2 = ep.Tt4_Tt2*onesv









    def set_baseline_params(self,segment_length):

        odp = self.offdesign_params
        bp = np.zeros([8,segment_length])
        # set the baseline params from the odp array
        bp[0,:] = odp.pi_f[0,:]
        bp[1,:] = odp.pi_lc[0,:]
        bp[2,:] = odp.pi_hc[0,:]
        bp[3,:] = odp.mf[0,:]
        bp[4,:] = odp.mlc[0,:]
        bp[5,:] = odp.mhc[0,:]
        bp[6,:] = odp.Tt4[0,:]
        bp[7,:] = odp.Pt5[0,:]

        return bp





    def update_engine_params(self,baseline_params):

        bp  = baseline_params
        odp = self.offdesign_params
        # set the engine params from a new base line
        odp.pi_f[0,:]  = bp[0,:]
        odp.pi_lc[0,:] = bp[1,:]
        odp.pi_hc[0,:] = bp[2,:]
        odp.mf[0,:]    = bp[3,:]
        odp.mlc[0,:]   = bp[4,:]
        odp.mhc[0,:]   = bp[5,:]
        odp.Tt4[0,:]   = bp[6,:]
        odp.Pt5[0,:]   = bp[7,:]


        return




    def jacobian(self,conditions,baseline_parameters,base_residuals):

        delta = 1e-8
        R     = base_residuals

        # Engine parameters (ep)
        bp = baseline_parameters
        ep = self.offdesign_params

        segment_length = len(conditions.propulsion.throttle)
        jacobian       = np.zeros([8,8,segment_length])

        # Network variable values
        network_vars = [ep.pi_f, ep.pi_lc, ep.pi_hc, ep.mf, ep.mlc, ep.mhc, ep.Tt4, ep.Pt5]

        design_run = False

        for i, network_var in enumerate(network_vars):
                network_var[0,:]         = bp[i,:]*(1.+delta)
                results                  = self.evaluate(conditions, design_run, ep)
                network_var[0,:]         = bp[i,:]
                jacobian[i,:,:]          = (results.Res - R)/(bp[i,:]*delta)

        jacobian = np.swapaxes(jacobian, 0, 1)


        return jacobian



    def engine_out(self,state):

        # Temporary throttle to save segment values
        temp_throttle = np.zeros(len(state.conditions.propulsion.throttle))

        # These for loops should be removed
        for i in range(0,len(state.conditions.propulsion.throttle)):
            temp_throttle[i] = state.conditions.propulsion.throttle[i]
            state.conditions.propulsion.throttle[i] = 1.0

        results = self.evaluate_thrust(state)

        for i in range(0,len(state.conditions.propulsion.throttle)):
            state.conditions.propulsion.throttle[i] = temp_throttle[i]

        results.thrust_force_vector = results.thrust_force_vector/self.number_of_engines*(self.number_of_engines-1)
        results.vehicle_mass_rate   = results.vehicle_mass_rate/self.number_of_engines*(self.number_of_engines-1)

        return results



    __call__ = evaluate_thrust

