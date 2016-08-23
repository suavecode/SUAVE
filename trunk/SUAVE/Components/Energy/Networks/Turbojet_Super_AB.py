# Turbojet_Super.py
#
# Created:  Nov 2015, Anil,Variyar,Tim MacDonald
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE
import copy

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


from SUAVE.Core import Data
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components.Propulsors.Propulsor import Propulsor


# ----------------------------------------------------------------------
#  Turbojet Network
# ----------------------------------------------------------------------

class Turbojet_Super_AB(Propulsor):

    def __defaults__(self):

        #setting the default values
        self.tag = 'Turbojet_AB'
        self.number_of_engines = 1.0
        self.nacelle_diameter  = 1.0
        self.engine_length     = 1.0

    _component_root_map = None


    # linking the different network components
    def evaluate_design(self,state):

        #Inputs M0,T0,gamma_c,c_pc,gamma_t,c_pt,h_PR,pi_dmax,pi_b,pi_n,e_c,e_t,eta_b,eta_m,P0_P9,Tt4,pi_c

        #Outputs F_mdpt0,f,S,eta_T,eta_Peta_O,eta_C,eta_t

        #imports
        conditions = state.conditions
        numerics   = state.numerics

        #engine properties
        pi_d_max = self.inlet_nozzle.pressure_ratio


        pi_cL = self.low_pressure_compressor.pressure_ratio
        pi_cH = self.high_pressure_compressor.pressure_ratio
        pi_c = pi_cL*pi_cH
        e_c = self.high_pressure_compressor.polytropic_efficiency

        e_cH = self.high_pressure_compressor.polytropic_efficiency
        e_cL = self.low_pressure_compressor.polytropic_efficiency


        eta_b = self.combustor.efficiency
        h_PR  = self.combustor.fuel_data.specific_energy
        pi_b  = self.combustor.pressure_ratio
        Tt4 = self.combustor.turbine_inlet_temperature


        eta_AB = self.afterburner.efficiency
        #h_PR  = self.afterburner.fuel_data.specific_energy
        pi_AB  = self.afterburner.pressure_ratio
        Tt7    = self.afterburner.afterburner_exit_temperature
        afterburner_on = self.afterburner.design


        eta_m = self.high_pressure_turbine.mechanical_efficiency
        e_t   = self.high_pressure_turbine.polytropic_efficiency

        e_tH = self.high_pressure_turbine.polytropic_efficiency
        e_tL = self.low_pressure_turbine.polytropic_efficiency

        P0_P9 = 1.0


        pi_n = self.core_nozzle.pressure_ratio

        sizing_thrust = self.thrust.total_design

        #gas properties
        gamma_c = 1.4
        gamma_t = 1.3
        gamma_AB = 1.4
        c_pc = 1004.5
        c_pt = 1004.5
        c_pAB = 1004.5
        gc  = 1.0

        #freestream properties
        T0 = conditions.freestream.temperature
        p0 = conditions.freestream.pressure
        M0 = conditions.freestream.mach_number
        a0 = conditions.freestream.speed_of_sound

        #Creating the network by manually linking the different components
        #single spool


        R_c = (gamma_c - 1.0)/gamma_c*c_pc
        R_t = (gamma_t - 1.0)/gamma_t*c_pt
        a0 = np.sqrt(gamma_c*R_c*gc*T0)

        V0 = a0*M0

        tau_r = 1.0 + 0.5*(gamma_c - 1.0)*(M0**2.0)
        pi_r = tau_r**(gamma_c/(gamma_c-1.0))

        eta_r = 1.0

        if(M0>1.0):
            eta_r = 1.0 - 0.075*((M0-1.0)**1.35)

        pi_d = pi_d_max*eta_r
        tau_lamda = c_pt*Tt4/(c_pc*T0)

        #tau_c = pi_c**((gamma_c-1.0)/(gamma_c*e_c))
        #eta_c = (pi_c**((gamma_c-1.0)/gamma_c)-1.0)/(tau_c-1.0)

        #lpc
        tau_cL = pi_cL**((gamma_c-1.0)/(gamma_c*e_cL))
        eta_cL = (pi_cL**((gamma_c-1.0)/gamma_c)-1.0)/(tau_cL-1.0)

        #hpc
        tau_cH = pi_cH**((gamma_c-1.0)/(gamma_c*e_cH))
        eta_cH = (pi_cH**((gamma_c-1.0)/gamma_c)-1.0)/(tau_cH-1.0)


        #combustor
        f = (tau_lamda - tau_r*tau_cL*tau_cH)/((h_PR*eta_b)/(c_pc*T0)-tau_lamda)

        #tau_t = 1.0 - (1.0/(eta_m*(1.0+f)))*tau_r*(tau_c-1.0)/tau_lamda
        #pi_t = tau_t**(gamma_t/((gamma_t-1.0)*e_t))
        #eta_t = (1.0-tau_t)/(1.0-tau_t**(1.0/e_t))

        #HPT
        tau_tH = 1.0 - (1.0/(eta_m*(1.0+f)))*tau_r*(tau_cH-1.0)/tau_lamda
        pi_tH = tau_tH**(gamma_t/((gamma_t-1.0)*e_tH))
        eta_tH = (1.0-tau_tH)/(1.0-tau_tH**(1.0/e_tH))

        #LPT
        tau_tL = 1.0 - (1.0/(eta_m*(1.0+f)))*tau_r*(tau_cL-1.0)/(tau_lamda*tau_tH)
        pi_tL = tau_tL**(gamma_t/((gamma_t-1.0)*e_tL))
        eta_tL = (1.0-tau_tL)/(1.0-tau_tL**(1.0/e_tL))


        if (afterburner_on == 0):
            R_AB = R_t
            cp_AB = c_pt
            gamma_AB = gamma_t
            Tt7 = Tt4*tau_tH*tau_tL
            pi_AB = 1.0
            f_AB = 0.0

        else:

            #Afterburner
            R_AB = (gamma_AB-1.0)/(gamma_AB)*c_pAB
            tau_lamdaAB = cp_AB/c_pc*Tt7/T0
            f_AB = (1.0 + f)*(tau_lamdaAB - tau_lamda*tau_t)/(eta_AB*h_PR/(c_pc*T0)-tau_lamdaAB)




        Pt9_P9 = P0_P9*pi_r*pi_d*pi_c*pi_b*pi_tL*pi_tH*pi_AB*pi_n
        T9_T0 = (Tt7/T0)/((Pt9_P9)**((gamma_AB-1.0)/gamma_AB))
        M9 = np.sqrt(2.0/(gamma_AB-1.0)*((Pt9_P9)**((gamma_AB-1.0)/gamma_AB)-1.0))
        V9_a0 = M9*np.sqrt(gamma_AB*R_AB*T9_T0/(gamma_c*R_c))

        F_mdot0 = a0/gc*(1.0 + f + f_AB)*V9_a0 - M0 + (1.0 + f +f_AB)*R_AB/R_c*(T9_T0/V9_a0)*((1.0-P0_P9)/gamma_c)
        S = (f + f_AB)/(F_mdot0)*3600.0*2.20462/0.224809


        eta_P = 2.0*gc*V0*F_mdot0/(a0*a0*(1.0+f+f_AB)*V9_a0*V9_a0 - M0*M0)
        eta_T = (a0*a0*(1.0+f+f_AB)*V9_a0*V9_a0 - M0*M0)/(2.0*gc*(f+f_AB)*h_PR)
        eta_0 = eta_P*eta_T

        mdot_0 = 1.0



        #reference values store
        reference = Data()

        reference.M0R = M0
        reference.T0R = T0
        reference.P0R = p0

        reference.tau_rR = tau_r
        reference.pi_rR = pi_r
        reference.Tt4R = Tt4

        reference.pi_dR = pi_d
        reference.pi_cLR = pi_cL
        reference.pi_cHR = pi_cH

        reference.tau_cHR = tau_cH
        reference.tau_tLR = tau_tL
        reference.tau_cLR = tau_cL
        reference.tau_tHR = tau_tH

        reference.M9R = M9
        reference.mdot0R  = 1.0 #mdot0

        reference.tau_lamdaR = tau_lamda


        reference.pi_tH = pi_tH
        reference.pi_tL = pi_tL
        reference.pi_tLR = pi_tL


        reference.eta_cL = eta_cL
        reference.eta_cH = eta_cH
        reference.eta_tL = eta_tL
        reference.eta_tH  = eta_tH

        reference.P0_P9 = P0_P9


        self.reference = reference


        #getting the network outputs from the thrust outputs

        F            = F_mdot0*[1,0,0]
        mdot         = 1.0
        #Isp          = thrust.outputs.specific_impulse
        #output_power = thrust.outputs.power
        F_vec        = conditions.ones_row(3) * 0.0
        F_vec[:,0]   = F[:,0]
        F            = F_vec

        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
        results.sfc = S

        return results



    # linking the different network components
    def evaluate_offdesign(self,state):

        #Inputs M0,T0,gamma_c,c_pc,gamma_t,c_pt,h_PR,pi_dmax,pi_b,pi_n,e_c,e_t,eta_b,eta_m,P0_P9,Tt4,pi_c

        #Outputs F_mdpt0,f,S,eta_T,eta_Peta_O,eta_C,eta_t
        #imports
        conditions = state.conditions
        numerics   = state.numerics
        reference = self.reference
        throttle = conditions.propulsion.throttle
        number_of_engines         = self.number_of_engines

        #freestream properties
        afterburner_on = 0
        T0 = conditions.freestream.temperature
        p0 = conditions.freestream.pressure
        M0 = conditions.freestream.mach_number
        #a0 = conditions.freestream.speed_of_sound

        #engine properties
        Tt4 = self.combustor.turbine_inlet_temperature

        pi_d_max = self.inlet_nozzle.pressure_ratio
        pi_b  = self.combustor.pressure_ratio
        pi_tH  = reference.pi_tH
        pi_n = self.core_nozzle.pressure_ratio

        eta_AB = self.afterburner.efficiency
        #h_PR  = self.afterburner.fuel_data.specific_energy
        pi_AB  = self.afterburner.pressure_ratio
        Tt7    = self.afterburner.afterburner_exit_temperature

        tau_tH  = reference.tau_tHR

        eta_cL = reference.eta_cL
        eta_cH = reference.eta_cH
        eta_b = self.combustor.efficiency
        eta_mH = self.high_pressure_turbine.mechanical_efficiency
        eta_mL = self.low_pressure_turbine.mechanical_efficiency

        e_cH = self.high_pressure_compressor.polytropic_efficiency
        e_cL = self.low_pressure_compressor.polytropic_efficiency

        P0_P9 = 1.0
        #gas properties
        gamma_c = 1.4
        gamma_t = 1.33 #1.4
        gamma_AB = 1.4
        #c_pc = 0.24*778.16 #1004.5
        #c_pt = 0.276*778.16 #004.5
        #g_c  = 32.174

        c_pc = 1004.5
        c_pt = 1004.5
        c_pAB = 1004.5
        g_c  = 1.0


        h_PR  = self.combustor.fuel_data.specific_energy #18400.0 #self.combustor.fuel_data.specific_energy


        #reference conditions

        M0R  = reference.M0R
        T0R = reference.T0R
        P0R = reference.P0R

        tau_rR = reference.tau_rR
        pi_rR = reference.pi_rR

        Tt4R = reference.Tt4R

        pi_dR = reference.pi_dR
        pi_cLR = reference.pi_cLR
        pi_cHR = reference.pi_cHR
        pi_tL  = reference.pi_tL
        tau_cHR = reference.tau_cHR
        tau_tLR = reference.tau_tLR
        M9R = reference.M9R

        mdot0R = reference.mdot0R
        tau_cLR = reference.tau_cLR
        pi_tLR = reference.pi_tLR
        tau_lamdaR = reference.tau_lamdaR
        eta_tL = reference.eta_tL
        eta_tH = reference.eta_tH


        tau_tL = copy.deepcopy(tau_tLR)

        #Creating the network by manually linking the different components
        gc = 1.0
        R_c = (gamma_c - 1.0)/gamma_c*c_pc
        R_t = (gamma_t - 1.0)/gamma_t*c_pt
        a0 = np.sqrt(gamma_c*R_c*gc*T0)

        V0 = a0*M0

        tau_r = 1.0 + 0.5*(gamma_c - 1.0)*(M0**2.0)
        pi_r = tau_r**(gamma_c/(gamma_c-1.0))

        eta_r = np.ones(len(M0)) #1.0

        #if(M0>1.0):
            #eta_r = 1.0 - 0.075*((M0-1.0)**1.35)

        eta_r[M0 > 1.0] = 1.0 - 0.075*(M0[M0 > 1.0] - 1.0)**1.35

        pi_d = pi_d_max*eta_r

        #compute min allowable throttle

        tau_cL_i = pi_cLR**((gamma_c-1.0)/(gamma_c*e_cL))
        tau_cH_i = pi_cHR**((gamma_c-1.0)/(gamma_c*e_cH))
        min_Tt4 = tau_r*tau_cL_i*tau_cH_i*T0
        max_Tt4 = Tt4
        delta_Tt4 = max_Tt4 - min_Tt4

        min_Tt7 = max_Tt4 + 100.0
        max_Tt7 = Tt7

        Tt4 = min_Tt4 + throttle*delta_Tt4
        throttle_local = np.ones_like(throttle)

        print "throttle : ",throttle,(throttle.all()>1.0)

        #if(throttle.any()>1.0):
        if(throttle[0]>1.0):
            afterburner_on = 1
            Tt4 = min_Tt4 + throttle_local*delta_Tt4
            print "aftburner :",Tt4

        #Tt4[throttle>1.0] = min_Tt4 + throttle_local[throttle>1.0]*delta_Tt4


        #throttle between 1 and 2, add afterburners with afterburner temperature between min and max



        #low pressure compressure
        tau_cL = 1.0 + (Tt4/T0)/(Tt4R/T0R) * tau_rR/tau_r*(tau_cLR-1.0)
        pi_cL = (1.0 + eta_cL*(tau_cL - 1.0))**(gamma_c/(gamma_c-1.0))

        #high pressure compressure
        tau_cH = 1.0 + (Tt4/T0)/(Tt4R/T0R)*tau_rR*tau_cLR/(tau_r*tau_cL)*(tau_cHR-1.0)
        pi_cH = (1.0 + eta_cH*(tau_cH - 1.0))**(gamma_c/(gamma_c-1.0))


        tau_lamda = c_pt*Tt4/(c_pc*T0)

        f = (tau_lamda - tau_r*tau_cL*tau_cH)/((h_PR*eta_b)/(c_pc*T0)-tau_lamda)

        mdot0 = mdot0R* (p0*pi_r*pi_d*pi_cL*pi_cH)/(P0R*pi_rR*pi_dR*pi_cLR*pi_cHR)*np.sqrt(Tt4R/Tt4)


        if (afterburner_on == 0):
            R_AB = R_t
            cp_AB = c_pt
            gamma_AB = gamma_t
            Tt7 = Tt4*tau_tH*tau_tL
            pi_AB = 1.0
            f_AB = 0.0

        else:
            R_AB = (gamma_AB-1.0)/(gamma_AB)*c_pAB
            tau_lamdaAB = c_pAB*Tt7/(c_pc*T0)
            f_AB = (tau_lamdaAB - tau_lamda*tau_tH*tau_tL)/(eta_AB*h_PR/(c_pc*T0)-tau_lamdaAB)
            print "aftburner test ",c_pAB,Tt7,c_pc,T0



        #Afterburner


        Pt9_P9 = P0_P9*pi_r*pi_d*pi_cL*pi_cH*pi_b*pi_tH*pi_tL*pi_AB*pi_n   #pi_r*pi_d*pi_c*pi_b*pi_t*pi_AB*pi_n
        M9 = np.sqrt(2.0/(gamma_AB-1.0)*((Pt9_P9)**((gamma_AB-1.0)/gamma_AB)-1.0))
        T9_T0 = (Tt7/T0)/((Pt9_P9)**((gamma_AB-1.0)/gamma_AB))
        V9_a0 = M9*np.sqrt(gamma_AB*R_AB*T9_T0/(gamma_c*R_c))

        f_O = f + f_AB
        print throttle,f,f_AB,f

        F_mdot0 = a0/gc*(1.0 + f_O)*V9_a0 - M0 + (1.0 + f_O)*R_AB/R_c*(T9_T0/V9_a0)*((1.0-P0_P9)/gamma_c)

        F = mdot0*F_mdot0

        S = (f_O)/(F_mdot0)

        eta_T = (a0*a0*(1.0+f_O)*V9_a0*V9_a0 - M0*M0)/(2.0*gc*(f_O)*h_PR)
        eta_P = 2.0*gc*V0*F_mdot0/(a0*a0*(1.0+f_O)*V9_a0*V9_a0 - M0*M0)

        eta_0 = eta_P*eta_T


        #add speeds component
        #N_NR_LP =

        sfc = S*3600.0*2.20462/0.224809

        #getting the network outputs from the thrust outputs

        F            = F*[1,0,0]
        mdot         = mdot0
        #Isp          = thrust.outputs.specific_impulse
        #output_power = thrust.outputs.power
        F_vec        = conditions.ones_row(3) * 0.0
        F_vec[:,0]   = F[:,0]
        F            = F_vec

        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
        results.sfc = sfc

        return results


    def size(self,mach_number = None, altitude = None, delta_isa = 0, conditions = None):

        #Unpack components


        if(conditions):
            #use conditions
            pass

        else:
            #check if mach number and temperature are passed
            if(mach_number==None or altitude==None):

                #raise an error
                raise NameError('The sizing conditions require an altitude and a Mach number')

            else:
                #call the atmospheric model to get the conditions at the specified altitude
                atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
                p,T,rho,a,mu = atmosphere.compute_values(altitude,delta_isa)

                # setup conditions
                conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()



                # freestream conditions

                conditions.freestream.altitude           = np.atleast_1d(altitude)
                conditions.freestream.mach_number        = np.atleast_1d(mach_number)

                conditions.freestream.pressure           = np.atleast_1d(p)
                conditions.freestream.temperature        = np.atleast_1d(T)
                conditions.freestream.density            = np.atleast_1d(rho)
                conditions.freestream.dynamic_viscosity  = np.atleast_1d(mu)
                conditions.freestream.gravity            = np.atleast_1d(9.81)
                conditions.freestream.gamma              = np.atleast_1d(1.4)
                conditions.freestream.Cp                 = 1.4*287.87/(1.4-1)
                conditions.freestream.R                  = 287.87
                conditions.freestream.speed_of_sound     = np.atleast_1d(a)
                conditions.freestream.velocity           = conditions.freestream.mach_number * conditions.freestream.speed_of_sound

                # propulsion conditions
                conditions.propulsion.throttle           =  np.atleast_1d(1.0)


        state = Data()
        state.numerics = Data()
        state.conditions = conditions
        number_of_engines         = self.number_of_engines

        sizing_thrust = self.thrust.total_design/float(number_of_engines)

        results_nondim = self.evaluate_design(state)

        F_mdot0 = results_nondim.thrust_force_vector
        S = results_nondim.sfc

        mdot0 = sizing_thrust/F_mdot0

        self.reference.mdot0R = mdot0

        results = Data()

        results.thrust_force_vector = self.thrust.total_design
        results.vehicle_mass_rate   = mdot0
        results.sfc                 = S
        results.thrust_non_dim      = F_mdot0




        #compute sls conditions
        #call the atmospheric model to get the conditions at the specified altitude
        atmosphere_sls = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        p,T,rho,a,mu = atmosphere_sls.compute_values(0.0,0.0)

        # setup conditions
        conditions_sls = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()



        # freestream conditions

        conditions_sls.freestream.altitude           = np.atleast_1d(0.)
        conditions_sls.freestream.mach_number        = np.atleast_1d(0.01)

        conditions_sls.freestream.pressure           = np.atleast_1d(p)
        conditions_sls.freestream.temperature        = np.atleast_1d(T)
        conditions_sls.freestream.density            = np.atleast_1d(rho)
        conditions_sls.freestream.dynamic_viscosity  = np.atleast_1d(mu)
        conditions_sls.freestream.gravity            = np.atleast_1d(9.81)
        conditions_sls.freestream.gamma              = np.atleast_1d(1.4)
        conditions_sls.freestream.Cp                 = 1.4*287.87/(1.4-1)
        conditions_sls.freestream.R                  = 287.87
        conditions_sls.freestream.speed_of_sound     = np.atleast_1d(a)
        conditions_sls.freestream.velocity           = conditions_sls.freestream.mach_number * conditions_sls.freestream.speed_of_sound

        # propulsion conditions
        conditions_sls.propulsion.throttle           =  np.atleast_1d(1.0)

        state_sls = Data()
        state_sls.numerics = Data()
        state_sls.conditions = conditions_sls
        results_sls = self.evaluate_offdesign(state_sls)
        self.sealevel_static_thrust = results_sls.thrust_force_vector[0,0] / float(number_of_engines)

        self.sealevel_static_mass_flow = results_sls.vehicle_mass_rate[0,0] / float(number_of_engines)
        self.sealevel_static_sfc = 3600. * self.sealevel_static_mass_flow / 0.1019715 / self.sealevel_static_thrust


        return results





        #return





    __call__ = evaluate_offdesign

