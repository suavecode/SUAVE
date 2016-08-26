#Turbofan_Network.py
#
# Created:  Anil Variyar, Oct 2015
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
import copy
from SUAVE.Core import Units

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn


from SUAVE.Core import Data
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components.Propulsors.Propulsor import Propulsor


# ----------------------------------------------------------------------
#  Turbofan Network
# ----------------------------------------------------------------------

class Turbofan_3(Propulsor):

    def __defaults__(self):

        #setting the default values
        self.tag = 'Turbo_Fan'
        self.number_of_engines = 1.0
        self.nacelle_diameter  = 1.0
        self.engine_length     = 1.0
        self.bypass_ratio      = 1.0
        self.surrogate         = 0
        self.surrogate_model   = None
        self.map_scale         = None

    _component_root_map = None




    def engine_out(self,state):


        temp_throttle = np.zeros(len(state.conditions.propulsion.throttle))

        for i in range(0,len(state.conditions.propulsion.throttle)):
            temp_throttle[i] = state.conditions.propulsion.throttle[i]
            state.conditions.propulsion.throttle[i] = 1.0



        results = self.evaluate_thrust(state)

        for i in range(0,len(state.conditions.propulsion.throttle)):
            state.conditions.propulsion.throttle[i] = temp_throttle[i]



        results.thrust_force_vector = results.thrust_force_vector/self.number_of_engines*(self.number_of_engines-1)
        results.vehicle_mass_rate   = results.vehicle_mass_rate/self.number_of_engines*(self.number_of_engines-1)



        return results



    #Solve the offdesign system of equations






    def size(self,mach_number = None, altitude = None, delta_isa = 0, conditions = None):


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


        self.thrust.bypass_ratio = self.bypass_ratio

        state = Data()
        state.numerics = Data()
        state.conditions = conditions
        number_of_engines         = self.number_of_engines

        sizing_thrust = self.thrust.total_design/float(number_of_engines)

        results_nondim = self.engine_design(state)

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
        results_sls = self.evaluate_thrustt(state_sls,0.0)
        self.sealevel_static_thrust = results_sls.thrust_force_vector[0,0] / float(number_of_engines)

        self.sealevel_static_mass_flow = results_sls.vehicle_mass_rate[0,0] / float(number_of_engines)
        self.sealevel_static_sfc = 3600. * self.sealevel_static_mass_flow / 0.1019715 / self.sealevel_static_thrust

        #print "sls thrust ",self.sealevel_static_thrust

        return results







    def engine_design(self,state):

        #imports
        conditions = state.conditions
        numerics   = state.numerics

        #engine properties
        pi_d_max = self.inlet_nozzle.pressure_ratio

        Tt4 = self.combustor.turbine_inlet_temperature
        pi_cL = self.low_pressure_compressor.pressure_ratio
        pi_cH = self.high_pressure_compressor.pressure_ratio
        pi_c = pi_cL*pi_cH
        e_c = self.high_pressure_compressor.polytropic_efficiency

        e_cH = self.high_pressure_compressor.polytropic_efficiency
        e_cL = self.low_pressure_compressor.polytropic_efficiency

        pi_f = self.fan.pressure_ratio
        e_f = self.fan.polytropic_efficiency

        eta_b = self.combustor.efficiency
        h_pr  = self.combustor.fuel_data.specific_energy
        pi_b  = self.combustor.pressure_ratio

        eta_m = self.high_pressure_turbine.mechanical_efficiency
        e_t   = self.high_pressure_turbine.polytropic_efficiency

        e_tH = self.high_pressure_turbine.polytropic_efficiency
        e_tL = self.low_pressure_turbine.polytropic_efficiency


        aalpha = self.thrust.bypass_ratio

        pi_n = self.core_nozzle.pressure_ratio

        pi_fn = self.fan_nozzle.pressure_ratio

        sizing_thrust = self.thrust.total_design

        #gas properties
        gamma_c = 1.4
        gamma_t = 1.3
        c_pc = 1004.5
        c_pt = 1004.5
        g_c  = 1.0

        #freestream properties
        T0 = conditions.freestream.temperature
        p0 = conditions.freestream.pressure
        M0 = conditions.freestream.mach_number
        a0 = conditions.freestream.speed_of_sound

        #design run

        R_c = (gamma_c - 1.0)/gamma_c*c_pc

        R_t = (gamma_t - 1.0)/gamma_t*c_pt

        a0 = np.sqrt(gamma_c*R_c*g_c*T0)

        V0 = a0*M0

        tau_r = 1.0 + (gamma_c - 1.0)*0.5*(M0**2.0)

        pi_r = tau_r**(gamma_c/(gamma_c-1.0))

        eta_r = 1.0

        if(M0>1.0):
            eta_r = 1.0 - 0.075*(M0 - 1.0)**1.35

        pi_d = pi_d_max*eta_r

        tau_lamda = c_pt*Tt4/(c_pc*T0)






        #P0_P19 = 1.0

        #P0_P9 = P0_P19


        ##compressor
        #tau_c = pi_c**((gamma_c-1.0)/(gamma_c*e_c))
        #eta_c = (pi_c**((gamma_c-1.0)/gamma_c) - 1.0)/(tau_c -1.0)

        #fan
        tau_f = pi_f**((gamma_c-1.0)/(gamma_c*e_f))
        eta_f = (pi_f**((gamma_c-1.0)/gamma_c) - 1.0)/(tau_f -1.0)


        #lpc
        tau_cL = pi_cL**((gamma_c-1.0)/(gamma_c*e_cL))
        eta_cL = (pi_cL**((gamma_c-1.0)/gamma_c) - 1.0)/(tau_cL -1.0)

        #hpc
        tau_cH = pi_cH**((gamma_c-1.0)/(gamma_c*e_cH))
        eta_cH = (pi_cH**((gamma_c-1.0)/gamma_c) - 1.0)/(tau_cH -1.0)



        #combustor
        tau_c = tau_cH
        #f = (tau_lamda - tau_r*tau_cH*tau_cL)/((eta_b*h_pr/(c_pc*T0)) - tau_lamda)
        f = (tau_lamda - tau_f*tau_r*tau_cH*tau_cL)/((eta_b*h_pr/(c_pc*T0)) - tau_lamda)

        #turbine
        #tau_t = 1.0 - tau_r/(eta_m*tau_lamda*(1.0+f))*(tau_c - 1.0 + aalpha*(tau_f -1.0))
        #pi_t = tau_t **(gamma_t/(gamma_t - 1.0)*e_t)
        #eta_t = (1.0-tau_t)/(1.0 - tau_t**(1/e_t))

        #HPT
        tau_tH = 1.0 - tau_r*tau_cL/(eta_m*tau_lamda*(1.0+f))*(tau_cH - 1.0)
        pi_tH = tau_tH **(gamma_t/((gamma_t - 1.0)*e_tH))
        eta_tH = (1.0-tau_tH)/(1.0 - tau_tH**(1/e_tH))

        #LPT
        tau_tL = 1.0 - tau_r/(eta_m*tau_lamda*tau_tH*(1.0+f))*(tau_cL - 1.0 + aalpha*(tau_f -1.0))
        pi_tL = tau_tL **(gamma_t/((gamma_t - 1.0)*e_tL))
        eta_tL = (1.0-tau_tL)/(1.0 - tau_tL**(1/e_tL))



        #P0_P9 = ((0.5*(gamma_c + 1.0))**(gamma_c/(gamma_c-1.0)))/(pi_r*pi_d*pi_cL*pi_cH*pi_b*pi_tL*pi_tH*pi_n) #*pif
        P0_P9 = ((0.5*(gamma_c + 1.0))**(gamma_c/(gamma_c-1.0)))/(pi_r*pi_d*pi_cL*pi_cH*pi_b*pi_tL*pi_tH*pi_n*pi_f) #*pif

        if(P0_P9>1.0):
            P0_P9 = 1.0



        P0_P19 = ((0.5*(gamma_c + 1.0))**(gamma_c/(gamma_c-1.0)))/(pi_r*pi_d*pi_f*pi_fn)

        if(P0_P19>1.0):
            P0_P19 = 1.0


        #pt9_p9 = P0_P9*(pi_r*pi_d*pi_cL*pi_cH*pi_b*pi_tH*pi_tL*pi_n) #*pi_f
        pt9_p9 = P0_P9*(pi_r*pi_d*pi_cL*pi_cH*pi_b*pi_tH*pi_tL*pi_n*pi_f) #*pi_f

        M9 = np.sqrt(2.0/(gamma_t-1.0)*((pt9_p9)**((gamma_t-1.0)/gamma_t)-1.0))

        #T9_T0 = tau_lamda*tau_tL*tau_tH*c_pc/(pt9_p9**((gamma_t-1.0)/gamma_t)*c_pt)
        T9_T0 = tau_lamda*tau_tL*tau_tH*c_pc/(pt9_p9**((gamma_t-1.0)/gamma_t)*c_pt)

        V9_a0 = M9*np.sqrt(gamma_t*R_t*T9_T0/(gamma_c*R_c))

        pt19_p19 = P0_P19*pi_r*pi_d*pi_f*pi_fn

        M19 = np.sqrt(2.0/(gamma_c-1.0)*((pt19_p19)**((gamma_c-1.0)/gamma_c)-1.0))

        T19_T0 = tau_r*tau_f/(pt19_p19**((gamma_c-1.0)/gamma_c))

        V19_a0 = M19*np.sqrt(T19_T0)

        F_mdot0 = 1.0/(1.0+aalpha)*a0/g_c*( (1.0+f)*V9_a0 - M0 + (1.0+f)*R_t*T9_T0*(1-P0_P9)/(R_c*V9_a0*gamma_c)) + aalpha/(1.0 + aalpha)*a0/g_c*(V19_a0 - M0 + T19_T0*(1-P0_P19)/(V19_a0*gamma_c))

        S = f/((1.0+aalpha)*F_mdot0)*3600.0*2.20462/0.224809









        #thrust_ratio =

        #eta_P

        #eta_T

        #eta_o = eta_P *eta_T

        #size to get mdot



        #mdot0 = sizing_thrust/F_mdot0 #1.0


        #reference values store
        reference = Data()

        reference.M0R = M0
        reference.T0R = T0
        reference.P0R = p0

        reference.tau_rR = tau_r
        reference.pi_rR = pi_r
        reference.Tt4R = Tt4

        reference.pi_dR = pi_d
        reference.pi_fR = pi_f
        reference.pi_cLR = pi_cL
        reference.pi_cHR = pi_cH

        reference.tau_fR = tau_f
        reference.tau_cHR = tau_cH
        reference.tau_tLR = tau_tL
        reference.tau_cLR = tau_cL
        reference.tau_tHR = tau_tH

        reference.alphaR = aalpha
        reference.M9R = M9
        reference.M19R = M19
        reference.mdot0R  = 1.0 #mdot0

        reference.tau_lamdaR = tau_lamda


        reference.pi_tH = pi_tH
        reference.pi_tL = pi_tL
        reference.pi_tLR = pi_tL


        reference.eta_f = eta_f
        reference.eta_cL = eta_cL
        reference.eta_cH = eta_cH
        reference.eta_tL = eta_tL
        reference.eta_tH  = eta_tH

        reference.P0_P9 = P0_P9

        reference.P0_P19 = P0_P19

        self.reference = reference






        results = Data()
        results.thrust_force_vector = F_mdot0
        results.vehicle_mass_rate   = 1.0
        results.sfc                 = S


        return results









    def evaluate_thrust(self,state,engine_efficiency=None):

        #imports
        conditions = state.conditions
        numerics   = state.numerics
        reference = self.reference
        throttle = conditions.propulsion.throttle

        #freestream properties
        T0 = conditions.freestream.temperature
        p0 = conditions.freestream.pressure
        M0 = conditions.freestream.mach_number

        F = np.zeros([len(T0),3])
        mdot0 = np.zeros([len(T0),1])
        S  = np.zeros(len(T0))
        F_mdot0 = np.zeros(len(T0))


        # setup conditions
        conditions_eval = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()

        state_eval = Data()
        state_eval.numerics = Data()

        for ieval in range(0,len(M0)):

            # freestream conditions

            conditions_eval.freestream.altitude           = np.atleast_1d(10.)
            conditions_eval.freestream.mach_number        = np.atleast_1d(M0[ieval])

            conditions_eval.freestream.pressure           = np.atleast_1d(p0[ieval][0])
            conditions_eval.freestream.temperature        = np.atleast_1d(T0[ieval][0])
            #conditions_eval.freestream.density            = np.atleast_1d(rho)
            #conditions_eval.freestream.dynamic_viscosity  = np.atleast_1d(mu)
            #conditions_eval.freestream.gravity            = np.atleast_1d(9.81)
            #conditions_eval.freestream.gamma              = np.atleast_1d(1.4)
            #conditions_eval.freestream.Cp                 = 1.4*287.87/(1.4-1)
            #conditions_eval.freestream.R                  = 287.87
            #conditions_eval.freestream.speed_of_sound     = np.atleast_1d(a)
            #conditions_eval.freestream.velocity           = conditions_eval.freestream.mach_number * conditions_eval.freestream.speed_of_sound

            # propulsion conditions
            conditions_eval.propulsion.throttle           =  np.atleast_1d(throttle[ieval])


            state_eval.conditions = conditions_eval
            results_eval = self.evaluate_thrustt(state_eval,engine_efficiency)

            F[ieval][0] = results_eval.thrust_force_vector
            mdot0[ieval][0] = results_eval.vehicle_mass_rate
            S[ieval] = results_eval.sfc
            F_mdot0[ieval] = results_eval.thrust_non_dim




        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot0
        results.sfc                 = S
        results.thrust_non_dim      = F_mdot0
        results.offdesigndata = results_eval.offdesigndata


        return results


    def evaluate_thrustt(self,state,delta_Tt4,engine_efficiency=None):

        #imports
        conditions = state.conditions
        numerics   = state.numerics
        reference = self.reference
        throttle = conditions.propulsion.throttle
        number_of_engines         = self.number_of_engines

        #freestream properties
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
        pi_fn = self.fan_nozzle.pressure_ratio

        tau_tH  = reference.tau_tHR

        eta_f = reference.eta_f
        eta_cL = reference.eta_cL
        eta_cH = reference.eta_cH
        eta_b = self.combustor.efficiency
        eta_mH = self.high_pressure_turbine.mechanical_efficiency
        eta_mL = self.low_pressure_turbine.mechanical_efficiency

        e_cH = self.high_pressure_compressor.polytropic_efficiency
        e_cL = self.low_pressure_compressor.polytropic_efficiency
        e_f = self.fan.polytropic_efficiency
        #eta_tL = reference.eta_tL
        #eta_tH = reference.eta_tH

        #gas properties
        gamma_c = 1.4
        gamma_t = 1.3 #1.4
        #c_pc = 0.24*778.16 #1004.5
        #c_pt = 0.276*778.16 #004.5
        #g_c  = 32.174

        c_pc = 1004.5
        c_pt = 1004.5
        g_c  = 1.0


        h_pr  = self.combustor.fuel_data.specific_energy #18400.0 #self.combustor.fuel_data.specific_energy


        #reference conditions

        M0R  = reference.M0R
        T0R = reference.T0R
        P0R = reference.P0R

        tau_rR = reference.tau_rR
        pi_rR = reference.pi_rR

        Tt4R = reference.Tt4R

        pi_dR = reference.pi_dR
        pi_fR = reference.pi_fR
        pi_cLR = reference.pi_cLR
        pi_cHR = reference.pi_cHR
        pi_tL  = reference.pi_tL
        tau_fR = reference.tau_fR
        #pi_f = reference.pi_fR
        tau_cHR = reference.tau_cHR
        tau_tLR = reference.tau_tLR
        alphaR = reference.alphaR
        M9R = reference.M9R
        M19R = reference.M19R

        mdot0R = reference.mdot0R
        tau_cLR = reference.tau_cLR
        pi_tLR = reference.pi_tLR
        tau_lamdaR = reference.tau_lamdaR
        eta_tL = reference.eta_tL
        eta_tH = reference.eta_tH
        #P0_P19 = reference.P0_P19
        #P0_P9  = reference.P0_P9
        #Tt4_min = self.combustor.Tt4_min
        #Tt4_max = self.combustor.Tt4_max


        #tau_fR = reference.tau_fR
        #tau_cLR = reference.tau_cLR
        #tau_tHR = reference.tau_tHR

        #tau_lamdaR = reference.tau_lamdaR
        #pi_tH = reference.pi_tH
        #pi_tL = reference.pi_tL


        #throttle to turbine inlet temperature

        #Tt4 = Tt4*throttle


        #Mattingly 8.52 a-j

        R_c = (gamma_c - 1.0)*c_pc/gamma_c
        R_t = (gamma_t - 1.0)*c_pt/gamma_t
        a0 = np.sqrt(gamma_c*R_c*g_c*T0)
        V0 = a0*M0
        tau_r = 1.0 + 0.5*(gamma_c-1.0)*(M0**2.)
        pi_r = tau_r**(gamma_c/(gamma_c-1.0))

        eta_r = np.ones(len(M0)) #1.0
        #if(M0 > 1.0):
        eta_r[M0 > 1.0] = 1.0 - 0.075*(M0[M0 > 1.0] - 1.0)**1.35

        pi_d = pi_d_max*eta_r

        #Tt4 = Tt4_min + throttle*Tt4_max

        tau_lamda = c_pt*Tt4/(c_pc*T0)


        #compurte the min allowable turbine temp

        #fan
        tau_f_i = pi_fR**((gamma_c-1.0)/(gamma_c*e_f))
        tau_cL_i = pi_cLR**((gamma_c-1.0)/(gamma_c*e_cL))
        tau_cH_i = pi_cHR**((gamma_c-1.0)/(gamma_c*e_cH))
        min_Tt4 = tau_r*tau_f_i*tau_cL_i*tau_cH_i*T0
        max_Tt4 = Tt4
        delta_Tt4 = max_Tt4 - min_Tt4



        Tt4 = Tt4 + delta_Tt4#*throttle #min_Tt4 + throttle*delta_Tt4

        #print "Tt4 : ",Tt4



        #initial values for iteration

        tau_tL = copy.deepcopy(tau_tLR)
        tau_fan = copy.deepcopy(tau_fR)
        tau_cL = copy.deepcopy(tau_cLR)
        #tau_tH  = tau_tHR


        pi_tL = pi_tLR
        pi_cL = pi_cLR


        #print "initial : ",alphaR,tau_fR

        tau_tL_prev = 1.0
        #8.57 a - o
        iteration = 0
        while (1):



            tau_cH = 1.0 + (Tt4/T0)/(Tt4R/T0R)*(tau_rR*tau_cLR*tau_fR)/(tau_r*tau_cL*tau_fan)*(tau_cHR-1.0)
            #tau_cH = 1.0 + (Tt4/T0)/(Tt4R/T0R)*(tau_rR*tau_cLR)/(tau_r*tau_cL)*(tau_cHR-1.0)

            pi_cH = (1.0 +eta_cH*(tau_cH-1.0))**(gamma_c/(gamma_c-1.0))

            pi_f = (1.0 + (tau_fan-1.0)*eta_f)**(gamma_c/(gamma_c-1.0))


            pt19_po = pi_r*pi_d*pi_f*pi_fn

            #pt19_p19 = (0.5*(gamma_c+1.0))**(gamma_c/(gamma_c-1.0))


            if(pt19_po<((0.5*(gamma_c+1.0))**(gamma_c/(gamma_c-1.0)))):

                pt19_p19 = pt19_po[0]

            else:

                pt19_p19 = (0.5*(gamma_c+1.0))**(gamma_c/(gamma_c-1.0))


            #pt19_p19[pt19_po<((0.5*(gamma_c+1.0))**(gamma_c/(gamma_c-1.0)))] = pt19_po[pt19_po<((0.5*(gamma_c+1.0))**(gamma_c/(gamma_c-1.0)))]


            M19 = np.sqrt(2.0/(gamma_c-1.0)*((pt19_p19)**((gamma_c-1.0)/gamma_c)-1.0))

            #pt9_po = pi_r*pi_d*pi_cL*pi_cH*pi_b*pi_tH*pi_tL*pi_n#*pi_f

            pt9_po = pi_r*pi_d*pi_cL*pi_cH*pi_b*pi_tH*pi_tL*pi_n*pi_f


            pt9_p9 = (0.5*(gamma_t+1.0))**(gamma_t/(gamma_t-1.0))

            #pt9_p9[pt9_po < ((0.5*(gamma_t+1.0))**(gamma_t/(gamma_t-1.0)))] = pt9_po[pt9_po < ((0.5*(gamma_t+1.0))**(gamma_t/(gamma_t-1.0)))]

            if(pt9_po < ((0.5*(gamma_t+1.0))**(gamma_t/(gamma_t-1.0)))):

                pt9_p9 = pt9_po[0]

            else:

                pt9_p9 = (0.5*(gamma_t+1.0))**(gamma_t/(gamma_t-1.0))



            M9 = np.sqrt(2.0/(gamma_t-1.0)*(((pt9_p9)**((gamma_t-1.0)/(gamma_t)))-1.0))


            mfp_m19 = MFP(M19,gamma_c,R_c,g_c)

            mfp_m19R = MFP(M19R,gamma_c,R_c,g_c) #0.5318 #MFP(M19R,gamma_c,R_c,g_c)


            #aalpha = alphaR*(pi_cLR*pi_cHR/pi_fR)/(pi_cL*pi_cH/pi_f)*np.sqrt((tau_lamda/(tau_r*tau_fan))/(tau_lamdaR/(tau_rR*tau_fR)))*mfp_m19/mfp_m19R

            aalpha = alphaR*(pi_cLR*pi_cHR/pi_fR)/(pi_cL*pi_cH/pi_f)*np.sqrt((tau_lamda/(tau_r*tau_fan))/(tau_lamdaR/(tau_rR*tau_fR)))*mfp_m19/mfp_m19R


            tau_fan = 1.0 + (tau_fR - 1.0)*((1.0-tau_tL)/(1.0-tau_tLR)*(tau_lamda/tau_r)/(tau_lamdaR/tau_rR)*(tau_cLR-1.0 + alphaR*(tau_fR-1.0))/(tau_cLR-1.0 + aalpha*(tau_fR-1.0)))

            tau_cL = 1.0 + (tau_fan-1.0)*(tau_cLR-1.0)/(tau_fR-1.0)

            pi_cL = (1.0 + eta_cL*(tau_cL-1.0))**(gamma_c/(gamma_c-1.0))

            tau_tL = 1.0 - eta_tL*(1.0-pi_tL**((gamma_t-1.0)/gamma_t))


            mfp_m9 = MFP(M9,gamma_t,R_t,g_c)

            mfp_m9R = MFP(M9R,gamma_t,R_t,g_c) #0.5224

            mfp_m9_ratio = mfp_m9R/mfp_m9

            pi_tL = pi_tLR*np.sqrt(tau_tL/tau_tLR)*mfp_m9_ratio


            iteration = iteration + 1


            temp_val = np.abs((tau_tL-tau_tL_prev)/tau_tL_prev)


            #print iteration,tau_tL,tau_fan,aalpha


            if(temp_val < 0.0001) or(iteration==20):
                break

            #if(iteration==10):
                #break

            tau_tL_prev = tau_tL





        #8.57a - 8.57 o


        mdot0 = mdot0R*(1+aalpha)/(1+alphaR)*(p0*pi_r*pi_d*pi_cL*pi_cH*pi_f)/(P0R*pi_rR*pi_dR*pi_cLR*pi_cHR*pi_fR)*np.sqrt(Tt4R/Tt4)

        #mdot0 = mdot0R*(1+aalpha)/(1+alphaR)*(p0*pi_r*pi_d*pi_cL*pi_cH)/(P0R*pi_rR*pi_dR*pi_cLR*pi_cHR)*np.sqrt(Tt4R/Tt4)



        #f = (tau_lamda - tau_r*tau_cL*tau_cH)/(h_pr*eta_b/(c_pc*T0) - tau_lamda)
        f = (tau_lamda - tau_fan*tau_r*tau_cL*tau_cH)/(h_pr*eta_b/(c_pc*T0) - tau_lamda)


        #Equation 8.52z - 8.52ag

        T9_T0 = (tau_lamda*tau_tH*tau_tL)/((pt9_p9)**((gamma_t-1.0)/gamma_t))*(c_pc/c_pt)

        V9_a0 = M9*np.sqrt(gamma_t*R_t*T9_T0/(gamma_c*R_c))

        T19_T0 = tau_r*tau_fan/((pt19_p19)**((gamma_c-1.0)/gamma_c))

        V19_a0 = M19*np.sqrt(T19_T0)

        #P0_P9 = ((0.5*(gamma_c + 1.0))**(gamma_c/(gamma_c-1.0)))/(pi_r*pi_d*pi_cL*pi_cH*pi_b*pi_tL*pi_tH*pi_n) #pif
        P0_P9 = ((0.5*(gamma_c + 1.0))**(gamma_c/(gamma_c-1.0)))/(pi_r*pi_d*pi_cL*pi_cH*pi_b*pi_tL*pi_tH*pi_n*pi_f) #pif

        #P0_P9[P0_P9>1.0] = 1.0


        if(P0_P9>1.0):
            P0_P9 = 1.0


        P0_P19 = ((0.5*(gamma_c + 1.0))**(gamma_c/(gamma_c-1.0)))/(pi_r*pi_d*pi_f*pi_fn)
        #P0_P19[P0_P19>1.0] = 1.0

        if(P0_P19>1.0):
            P0_P19 = 1.0


        F_mdot0 = 1.0/(1.0+aalpha)*a0/g_c*((1.0+f)*V9_a0 - M0 + (1.0+f)*R_t*T9_T0*(1-P0_P9)/(R_c*V9_a0*gamma_c)) + aalpha/(1.0+aalpha)*a0/g_c*(V19_a0 - M0 + T19_T0*(1-P0_P19)/(V19_a0*gamma_c))

        #print "F_mdot0",F_mdot0,mdot0,pi_f,aalpha

        S = f/((1+aalpha)*F_mdot0)*3600.0*2.20462/0.224809

        F = mdot0*F_mdot0

        N_NR_fan = np.sqrt((T0*tau_r)/(T0R*tau_rR)*(pi_f**((gamma_c-1.0)/gamma_c)-1.0)/(pi_fR**((gamma_c-1.0)/gamma_c)-1.0))


        N_NR_HP = np.sqrt((T0*tau_r*tau_cL/(T0R*tau_rR*tau_cLR)*(pi_cH**((gamma_c-1.0)/gamma_c)-1.0)/(pi_cHR**((gamma_c-1.0)/gamma_c)-1.0)))


        ##update the throttle equation
        #if( N_NR_HP/throttle < 1):
            #Tt4 = Tt4 + Tt4*N_NR_HP
        #elif (N_NR_HP/throttle > 1):
            #Tt4 = Tt4 - Tt4*N_NR_HP




        #Equation 8.52ai - 8.52ak

        eta_T = (a0**2.0)*((1.0+f)*(V9_a0**2.0) + aalpha*(V19_a0**2.0) - (1.0+aalpha)*M0**2.0)/(2.0*g_c*f*h_pr)

        eta_P = (2.0*g_c*V0*(1+aalpha)*F_mdot0)/((a0**2.0)*((1.0+f)*(V9_a0**2.0) + aalpha*(V19_a0**2.0) - (1.0+aalpha)*M0**2.0))

        eta_0 = eta_P*eta_T

        mdot_fuel = S*F*0.224808943/(2.20462*3600.)


        #print "number of engines : ",float(number_of_engines),F

        Tt2_t = tau_r*T0

        #print "condition : ",M0
        #print "thrust, sfc, eta_th, eta_prop, eta_overall, BPR, FPR, LPC, HPC, HPT, LPT, Tt4, Tt4/Tt2, mdot_air"
        #print F,S,eta_T,eta_P,eta_0,aalpha,pi_f,pi_cL,pi_cH,1.0/pi_tH,1.0/pi_tL,Tt4,Tt4/Tt2_t,mdot0

        Tt3 = T0*tau_r*tau_cL*tau_cH
        pt3 = p0*pi_cL*pi_cH

        offdesigndata = Data()

        offdesigndata.F = F
        offdesigndata.S = S
        offdesigndata.eta_T = eta_T
        offdesigndata.eta_P = eta_P
        offdesigndata.aalpha = aalpha
        offdesigndata.pi_f = pi_f
        offdesigndata.pi_cL = pi_cL
        offdesigndata.pi_cH = pi_cH
        offdesigndata.pi_tH = 1.0/ pi_tH
        offdesigndata.pi_tL = 1.0/pi_tL
        offdesigndata.Tt4 = Tt4
        offdesigndata.Tt4_Tt2 = Tt4/Tt2_t
        offdesigndata.mdot0 = mdot0
        offdesigndata.mdotf = mdot_fuel
        offdesigndata.Tt3 = Tt3
        offdesigndata.pt3 = pt3



        results = Data()
        results.thrust_force_vector = F*float(number_of_engines)#*throttle
        results.vehicle_mass_rate   = mdot_fuel*float(number_of_engines)#*throttle
        results.sfc                 = S
        results.thrust_non_dim      = F_mdot0
        results.offdesigndata       = offdesigndata
        results.N_HP                = N_NR_HP




        return results






    __call__ = evaluate_thrust #evaluate_thrust



def MFP(M,gamma,R,g_c):

    #gamma = 1.4
    #g_c = 1.0
    #R = 287.0
    mfp_val = np.sqrt(gamma*g_c/R)*M/((1.0+ 0.5*(gamma-1.0)*M*M)**(0.5*(gamma+1.0)/(gamma-1.0)))


    #mfp_val = gamma*M/(((1.0+ 0.5*(gamma-1.0)*M*M)**(0.5*(gamma+1.0)/(gamma-1.0)))*np.sqrt(gamma*g_c/R))


    return mfp_val