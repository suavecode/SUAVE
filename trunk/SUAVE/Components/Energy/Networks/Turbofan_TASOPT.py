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
#from Turbofan_Jacobian import Turbofan_Jacobian


# ----------------------------------------------------------------------
#  Turbofan Network
# ----------------------------------------------------------------------

class Turbofan_TASOPT(Propulsor):

    def __defaults__(self):

        #setting the default values
        self.tag = 'Turbo_Fan'
        self.number_of_engines = 1.0
        self.nacelle_diameter  = 1.0
        self.engine_length     = 1.0
        self.bypass_ratio      = 1.0

        self.design_params     = None
        self.offdesign_params  = None
        self.max_iters         = 80 #1
        self.newton_relaxation = 1.0 #0.8*np.ones(8)
        #self.newton_relaxation[6] = 1.0
        self.compressor_map_file = "Compressor_map.txt"
        self.cooling_flow = 0
        self.no_of_turbine_stages = 0
        #self.an_jacobian = None


    _component_root_map = None


    def unpack(self):

        self.design_params    = Data()
        self.offdesign_params = Data()
        #self.an_jacobian = Turbofan_Jacobian()

        dp    = self.design_params
        #odp   = self.offdesign_params


        #self.inlet_nozzle
        #self.low_pressure_compressor
        #self.high_pressure_compressor
        #self.fan
        #self.combustor
        #self.high_pressure_turbine
        #self.low_pressure_turbine
        #self.core_nozzle
        #self.fan_nozzle
        #self.thrust


        #design parameters---------------------------------------

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
        dp.Tt4_spec = np.copy(dp.Tt4) #1680.
        dp.N_spec   = 1.0
        dp.Tt4_Tt2  = 1.0

        #offdesign parameters---------------------------------------

        odp = copy.deepcopy(dp)
        self.offdesign_params = odp








    def evaluate(self,conditions,flag,dp_eval = None):

        #Unpack

        if dp_eval:
            dp = dp_eval
        else:
            dp = self.offdesign_params


        P0 = conditions.freestream.pressure.T
        T0 = conditions.freestream.temperature.T
        M0  = conditions.freestream.mach_number.T
        gamma = 1.4
        Cp    = 1.4*287.87/(1.4-1)
        R     = 287.87
        g     = 9.81
        throttle = conditions.propulsion.throttle
        #throttle = 0.6 + conditions.propulsion.throttle*(1.0-0.6)/1.0
        #throttle[throttle<=0.6] = 0.6
        #throttle[throttle>1.1]  = 1.1
        N_spec  = throttle.T
        dp.N_spec = throttle.T
        results = Data()
        #dNfn_pif = 1.0
        #dNln_pilc = 1.0
        #dNfn_mf = 1.0
        #dNln_mlc = 1.0

        if(flag == 1):
            #lpc
            Nln,dp.eta_lc,dNln_pilc,dNln_mlc = self.performance(dp.pi_lc,dp.mlc,0)

            #hpc
            Nhn,dp.eta_hc,dNhn_pihc,dNhn_mhc = self.performance(dp.pi_hc,dp.mhc,1)

            #fpc
            Nfn,dp.eta_f,dNfn_pif,dNfn_mf = self.performance(dp.pi_f,dp.mf,2)



        a0 = np.sqrt(gamma*R*T0)
        u0 = M0*a0

        #freestream stagnation properties
        Tt0 = Tt(M0, T0, gamma)
        Pt0 = Pt(M0, P0, gamma)
        ht0 = Cp*Tt0
        h0  = Cp*T0

        #inlet conditions
        Pt1_8 = Pt0*dp.pi_d
        Tt1_8 = Tt0
        ht1_8 = ht0

        Pt1_9 = Pt1_8
        Tt1_9 = Tt1_8
        ht1_9 = ht1_8

        #fan and lpc inlet
        Pt2 = Pt1_9
        Tt2 = Tt1_9
        ht2 = ht1_9


        #fan
        Pt2_1 = Pt2*dp.pi_f
        Tt2_1 = Tt2*dp.pi_f**((gamma-1.)/(gamma*dp.eta_f))
        ht2_1 = Cp*Tt2_1

        #fan nozzle
        Pt7 = Pt2_1*dp.pi_fn
        Tt7 = Tt2_1*dp.pi_fn**((gamma-1.)*dp.eta_fn/(gamma))
        ht7 = Cp*Tt7

        #low pressure compressor
        Pt2_5 = Pt2*dp.pi_lc
        Tt2_5 = Tt2*dp.pi_lc**((gamma-1.)/(gamma*dp.eta_lc))
        #Pt2_5 = Pt2_1*dp.pi_lc
        #Tt2_5 = Tt2_1*dp.pi_lc**((gamma-1.)/(gamma*dp.eta_lc))
        ht2_5 = Cp*Tt2_5

        #high pressure compressor
        Pt3 = Pt2_5*dp.pi_hc
        Tt3 = Tt2_5*dp.pi_hc**((gamma-1.)/(gamma*dp.eta_hc))
        ht3 = Cp*Tt3

        #print " ",T0," ",P0," ",M0
        #print " ",Pt2," ",Tt2," ",Pt2_1," ",Tt2_1," ",Pt7," ",Tt7," ",Pt2_5," ",Tt2_5," ",Tt3," ",Pt3
        #print " ",dp.pi_lc," "," ",dp.pi_hc," "," ",dp.pi_f," "
        #if(flag==1):
            #print " ",dp.mlc," ",dp.mhc," ",dp.mf
            #print " ",Nln," ",dp.eta_lc," ",Nhn," ",dp.eta_hc," ",Nfn," ",dp.eta_f

        if(self.cooling_flow == 0):


            #combustor
            ht4  = Cp*dp.Tt4
            f    = (ht4 - ht3)/(dp.eta_b*dp.htf-ht4)
            Pt4  = Pt3*dp.pi_b

            Tt4_1 = 1.0*dp.Tt4
            Pt4_1 = Pt4
            ht4_1 = Cp*Tt4_1


        elif(self.cooling_flow == 1):

            theta_f = 0.4
            St_A    = 0.035
            eta_cf  = 0.7
            dTemp_streak = 200.0
            T_metal      = 1400.0
            M4a          = 0.8
            ruc          = 0.9

            Mexit = 0.8
            Tg_c = np.zeros(self.no_of_turbine_stages)
            theta_n = np.zeros(self.no_of_turbine_stages)
            cooling_massflow = np.zeros(self.no_of_turbine_stages)
            total_cooling_massflow = 0.


            Tg_c[0] = dp.Tt4 + dTemp_streak
            theta_n[0] = (Tg_c[0]-T_metal)/(Tg_c[0]-Tt3)
            cooling_massflow[0] = 1./(1. + (eta_cf/St_A*(1.-theta_n[0]))/(theta_n[0]*(1.0 - eta_cf*theta_f) - theta_f*(1.0 - eta_cf)))
            total_cooling_massflow = cooling_massflow[0]

            #computing the cooling mass flow

            #for ieng in range(1,self.no_of_turbine_stages):
                #Tg_c[ieng] = dp.Tt4*((1. + 0.2*(Mexit**2.0))**(-ieng))
                #theta_n[ieng] = (Tg_c[ieng]-T_metal)/(Tg_c[ieng]-Tt3)
                #cooling_massflow[ieng] = 1/(1. + (eta_cf/St_A*(1-theta_n[ieng]))/(theta_n[ieng]*(1.0 - eta_cf*theta_f) - theta_f*(1.0 - eta_cf)))
                #total_cooling_massflow += cooling_massflow[ieng]



            #combustor
            ht4  = Cp*dp.Tt4
            f    = (ht4 - ht3)*(1.-total_cooling_massflow)/(dp.eta_b*dp.htf-ht4)
            Pt4  = Pt3*dp.pi_b

            #compute the 4_1 conditions
            Tt4_1 = (dp.htf*f/Cp + Tt3)/(1. + f)
            u4a = M4a*np.sqrt(1.4*287*dp.Tt4)/(np.sqrt(1. + 0.2*M4a*M4a))
            uc = ruc*u4a

            u4_1 = ((1.0-total_cooling_massflow + f)*u4a + total_cooling_massflow*uc)/(1.0+f)
            T4_1 = Tt4_1 - 0.5*u4_1*u4_1/Cp

            P4_1 = Pt4*((1.0 + 0.2*M4a*M4a)**(-1.4/(0.4)))

            Pt4_1 = P4_1*((Tt4_1/T4_1)**(1.4/0.4))

            ht4_1 = Cp*Tt4_1

            #print




        #update the bypass ratio
        if (flag == 1):
            mcore = dp.mlc*(Pt2/dp.Pref)/np.sqrt(Tt2/dp.Tref)/(1.0 + f)
            #mcore = dp.mlc*(Pt2_1/dp.Pref)/np.sqrt(Tt2_1/dp.Tref)/(1.0 + f)
            mfan  = dp.mf*(Pt2/dp.Pref)/np.sqrt(Tt2/dp.Tref)/(1.0 + f)
            dp.aalpha = mfan/mcore

            #mlcD = (1.0+f)*mdot_core*np.sqrt(Tt2/dp.Tref)/(Pt2/dp.Pref)
            #mfD  = (1.0+f)*dp.aalpha*mdot_core*np.sqrt(Tt2/dp.Tref)/(Pt2/dp.Pref)




        #high pressure turbine
        deltah_ht = -1./(1.+f)*1./dp.etam_ht*(ht3-ht2_5)
        Tt4_5     = Tt4_1 + deltah_ht/Cp
        Pt4_5     = Pt4_1*(Tt4_5/Tt4_1)**(gamma/((gamma-1.)*dp.eta_ht))
        ht4_5     = ht4_1 + deltah_ht



        if(flag == 0): #design iteration

            #low pressure turbine
            deltah_lt =  -1./(1.+f)*1./dp.etam_lt*((ht2_5 - ht1_9)+ dp.aalpha*(ht2_1 - ht2))
            #deltah_lt =  -1./(1.+f)*1./dp.etam_lt*((ht2_5 - ht2_1)+ dp.aalpha*(ht2_1 - ht2))
            Tt4_9     = Tt4_5 + deltah_lt/Cp
            Pt4_9     = Pt4_5*(Tt4_9/Tt4_5)**(gamma/((gamma-1.)*dp.eta_lt))
            ht4_9     = ht4_5 + deltah_lt
            #Pt5       = Pt4_9*dp.pi_tn


        else:

            pi_lt = 1.0/dp.pi_tn*(dp.Pt5/Pt4_5)
            Pt4_9 = Pt4_5*pi_lt
            Tt4_9 = Tt4_5*pi_lt**((gamma-1.)*dp.eta_lt/(gamma))
            ht4_9 = Cp*Tt4_9
            #Pt5   = dp.Pt5



        #print" aalpha : ",dp.Tt4," ",Tt4_1," ",f," ",dp.aalpha," ",Tt4_5," ",Pt4_5," ",Tt4_9," ",Pt4_9


        #turbine nozzle
        Pt5 = Pt4_9*dp.pi_tn
        Tt5 = Tt4_9
        ht5 = ht4_9


        #core exhaust
        Pt6 = Pt5
        Tt6 = Tt5
        ht6 = Cp*Tt6

        P6 = Pt6*(P0/Pt6)
        T6 = Tt6*(P0/Pt6)**((gamma-1.)/(gamma))
        h6 = Cp*T6

        if(Tt6<T6):
            M6 = 1.0
            T6 = Tt6/(1+0.2*M6**2.0)
            u6 = M6*np.sqrt(gamma*R*T6)
        else:
            u6 = np.sqrt(2.0*(ht6-h6))


        #fan exhaust
        Pt8 = Pt7
        Tt8 = Tt7
        ht8 = Cp*Tt7

        P8 = Pt8*(P0/Pt8)
        T8 = Tt8*(P0/Pt8)**((gamma-1.)/(gamma))
        h8 = Cp*T8
        u8 = np.sqrt(2.0*(ht8-h8))



        #overall quantities
        Fsp = ((1.+f)*u6 - u0 + dp.aalpha*(u8-u0))/((1.+dp.aalpha)*a0)
        Isp = Fsp/f*a0/g*(1.0+dp.aalpha)

        sfc = 3600./Isp  #1./Isp

        #print " Fsp : ",Fsp," ",u8," ",Pt8," ",Tt8," ",Pt6," ",Tt6," ",u6

        if(flag==0):

            #run sizing analysis
            FD = self.thrust.total_design/(self.number_of_engines)
            F    = Fsp
            mdot = 1.0


            #core mass flow computation
            mdot_core = FD/(Fsp*a0*(1.+dp.aalpha))

            #fan area sizing
            T2 = Tt_inv(dp.M2, Tt2, gamma)
            P2 = Pt_inv(dp.M2, Pt2, gamma)
            h2 = Cp*T2
            rho2 = P2/(R*T2)
            u2 = dp.M2*np.sqrt(gamma*R*T2)
            A2 = (1.+dp.aalpha)*mdot_core/(rho2*u2)

            df2 = np.sqrt(4.0/np.pi*A2/(1.0-dp.HTR_f**2.0))



            #HP compressor area
            T2_5 = Tt_inv(dp.M2_5, Tt2_5, gamma)
            P2_5 = Pt_inv(dp.M2_5, Pt2_5, gamma)
            h2_5 = Cp*T2_5
            rho2_5 = P2_5/(R*T2_5)
            u2_5 = dp.M2_5*np.sqrt(gamma*R*T2_5)
            A2_5 = (1.+dp.aalpha)*mdot_core/(rho2_5*u2_5)

            df2_5 = np.sqrt(4./np.pi*A2_5/(1.0-dp.HTR_hc**2.0))


            #fan nozzle area
            M8 = u8/np.sqrt(gamma*R*T8)

            if(M8<1):
                P7 = P0
                M7 = np.sqrt((((Pt7/P7)**((gamma-1.)/gamma))-1.)*2./(gamma-1.))

            else:
                M7 = 1.0
                P7 = Pt7/(1.+(gamma-1.)/2.*M7**2.)**(gamma/(gamma-1.))



            T7 = Tt7/(1.+(gamma-1.)/2.*M7**2.)
            h7 = Cp*T7
            u7 = np.sqrt(2.0*(ht7-h7))
            rho7 = P7/(R*T7)
            A7   = dp.aalpha*mdot_core/(rho7*u7)


            #core nozzle area
            M6 = u6/np.sqrt(gamma*R*T6)

            if(M6<1.):
                P5 = P0
                M5 = np.sqrt((((Pt5/P5)**((gamma-1.)/gamma))-1.)*2./(gamma-1.))

            else:
                M5 = 1.0
                P5 = Pt5/(1.+(gamma-1.)/2.*M5**2.)**(gamma/(gamma-1.))



            T5 = Tt5/(1.+(gamma-1.)/2.*M5**2.)
            h5 = Cp*T5
            u5 = np.sqrt(2.0*(ht5-h5))
            rho5 = P5/(R*T5)
            A5   = mdot_core/(rho5*u5)



            #spool speed

            NlcD = 1.0
            NhcD = 1.0

            #non dimensionalization

            NlD = NlcD*1.0/np.sqrt(Tt1_9/dp.Tref)
            #NlD = NlcD*1.0/np.sqrt(Tt2_1/dp.Tref)
            NhD = NhcD*1.0/np.sqrt(Tt2_5/dp.Tref)

            mhtD = (1.0+f)*mdot_core*np.sqrt(Tt4_1/dp.Tref)/(Pt4_1/dp.Pref)
            mltD = (1.0+f)*mdot_core*np.sqrt(Tt4_5/dp.Tref)/(Pt4_5/dp.Pref)

            mhcD = (1.0+f)*mdot_core*np.sqrt(Tt2_5/dp.Tref)/(Pt2_5/dp.Pref)
            mlcD = (1.0+f)*mdot_core*np.sqrt(Tt2/dp.Tref)/(Pt2/dp.Pref)
            #mlcD = (1.0+f)*mdot_core*np.sqrt(Tt2_1/dp.Tref)/(Pt2_1/dp.Pref)
            mfD  = (1.0+f)*dp.aalpha*mdot_core*np.sqrt(Tt2/dp.Tref)/(Pt2/dp.Pref)


            dpp = self.design_params

            dpp.A5   = A5
            dpp.A7   = A7
            dpp.A2   = A2
            dpp.A2_5 = A2_5


            dpp.mhtD = mhtD
            dpp.mltD = mltD

            dpp.mhcD = mhcD
            dpp.mlcD = mlcD
            dpp.mfD  = mfD

            dpp.NlcD = NlcD
            dpp.NhcD = NhcD
            dpp.NlD  = NlD
            dpp.NhD  = NhD

            dpp.mhtD = mhtD
            dpp.mltD = mltD


            #update the offdesign params
            dpp.mhc = mhcD
            dpp.mlc = mlcD
            dpp.mf  = mfD

            dpp.Nl = NlcD
            dpp.Nh = NhcD
            dpp.Nf  = 1.0

            dpp.Nln  = NlD
            dpp.Nhn  = NhD
            dpp.Nfn  = NlD

            dpp.Pt5  = Pt5
            dpp.Tt4_Tt2 = dp.Tt4/Tt2

            #print







        else:

            dp.Tt3 = Tt3
            dp.Pt3 = Pt3
            ##lpc
            #Nln,dp.eta_lc = self.performance(dp.pi_lc,dp.mlc,0)
            dp.Nl = np.sqrt(Tt2/dp.Tref)*Nln
            #dp.Nl = np.sqrt(Tt2_1/dp.Tref)*Nln

            ##hpc
            #Nhn,dp.eta_hc = self.performance(dp.pi_hc,dp.mhc,1)
            dp.Nh = np.sqrt(Tt2_5/dp.Tref)*Nhn

            ##fpc
            #Nfn,dp.eta_f = self.performance(dp.pi_f,dp.mf,2)
            dp.Nf = np.sqrt(Tt1_9/dp.Tref)*Nfn


            if(flag == 1):
                dp.Tt4_spec = dp.Tt4_Tt2*Tt2*(throttle)
                #dp.Tt4_spec = 2000.


            mdot_core = dp.mlc*Pt2/dp.Pref/np.sqrt(Tt2/dp.Tref)
            #mdot_core = dp.mlc*Pt2_1/dp.Pref/np.sqrt(Tt2_1/dp.Tref)
            Fsp = ((1.+f)*u6 - u0 + dp.aalpha*(u8-u0))/((1.+dp.aalpha)*a0)

            Isp  = Fsp*a0*(1.+dp.aalpha)/(f*g)
            TSFC = 3600.0/(Isp)
            F    = Fsp*(1+dp.aalpha)*mdot_core*a0
            mdot = mdot_core*f

            dp.mdot_core = mdot_core
            #compute dp.M2

            #fan area sizing
            T2 = Tt_inv(dp.M2, Tt2, gamma)
            P2 = Pt_inv(dp.M2, Pt2, gamma)
            h2 = Cp*T2
            rho2 = P2/(R*T2)
            u2 = dp.M2*np.sqrt(gamma*R*T2)

            #print P0.shape,T2.shape

            #HP compressor area
            T2_5 = Tt_inv(dp.M2_5, Tt2_5, gamma)
            P2_5 = Pt_inv(dp.M2_5, Pt2_5, gamma)
            h2_5 = Cp*T2_5
            rho2_5 = P2_5/(R*T2_5)
            u2_5 = dp.M2_5*np.sqrt(gamma*R*T2_5)

            #print " mdot : ",mdot_core," ",Fsp," ",Isp," ",F," ",mdot," ",u2," ",u2_5

            #fan nozzle area
            M8 = u8/np.sqrt(gamma*R*T8)

            P7 = np.zeros(P0.shape)
            M7 = np.zeros(M0.shape)
            #if(M8<1):
            P7[M8<1.] = P0[M8<1]
            M7[M8<1.] = (np.sqrt((((Pt7/P7)**((gamma-1.)/gamma))-1.)*2./(gamma-1.)))[M8<1]

            #else:
            M7[M8>=1.] = 1.0
            P7[M8>=1.]  = (Pt7/(1.+(gamma-1.)/2.*M7**2.)**(gamma/(gamma-1.)))[M8>=1]



            T7 = Tt7/(1.+(gamma-1.)/2.*M7**2.)
            h7 = Cp*T7
            u7 = np.sqrt(2.0*(ht7-h7))
            rho7 = P7/(R*T7)


            #core nozzle area
            M6 = u6/np.sqrt(gamma*R*T6)


            P5 = np.zeros(P0.shape)
            M5 = np.zeros(M0.shape)

            #print "M6 : ",M6," M6<1.0 : ",M6<1.0," M5 : ",M5[M6<1.0]

            #if(M6<1):
            P5[M6<1.] = P0[M6<1]
            M5[M6<1.] = (np.sqrt((((Pt5/P5)**((gamma-1.)/gamma))-1.)*2./(gamma-1.)))[M6<1]

            #else:
            M5[M6>=1.] = 1.0
            P5[M6>=1.] = (Pt5/(1.+(gamma-1.)/2.*M5**2.)**(gamma/(gamma-1.)))[M6>=1]



            T5 = Tt5/(1.+(gamma-1.)/2.*M5**2.)
            h5 = Cp*T5
            u5 = np.sqrt(2.0*(ht5-h5))
            rho5 = P5/(R*T5)




            #compute offdesign residuals


            Res = np.zeros([8,M0.shape[1]])

            Res[0,:] = (dp.Nf*dp.GF - dp.Nl)#/dp.Nl
            Res[1,:] = ((1.+f)*dp.mhc*np.sqrt(Tt4_1/Tt2_5)*Pt2_5/Pt4_1 - dp.mhtD)#/dp.mhtD
            Res[2,:] = ((1.+f)*dp.mhc*np.sqrt(Tt4_5/Tt2_5)*Pt2_5/Pt4_5 - dp.mltD)#/dp.mltD
            Res[3,:] = (dp.mf*np.sqrt(dp.Tref/Tt2)*Pt2/dp.Pref - rho7*u7*dp.A7)#/(rho7*u7*dp.A7)
            Res[4,:] = ((1.+f)*dp.mhc*np.sqrt(dp.Tref/Tt2_5)*Pt2_5/dp.Pref - rho5*u5*dp.A5)#/rho5*u5*dp.A5

            Res[5,:] = (dp.mlc*np.sqrt(dp.Tref/Tt1_9)*Pt1_9/dp.Pref - dp.mhc*np.sqrt(dp.Tref/Tt2_5)*Pt2_5/dp.Pref)#/(dp.mhc*np.sqrt(dp.Tref/Tt2_5)*Pt2_5/dp.Pref)
            #Res[5,:] = (dp.mlc*np.sqrt(dp.Tref/Tt2_1)*Pt2_1/dp.Pref - dp.mhc*np.sqrt(dp.Tref/Tt2_5)*Pt2_5/dp.Pref)#/(dp.mhc*np.sqrt(dp.Tref/Tt2_5)*Pt2_5/dp.Pref)

            #Res[6,:] = (dp.Tt4 - dp.Tt4_spec)#/dp.Tt4

            Res[6,:] = (dp.Nl - dp.N_spec)#/dp.Tt4
            #Res[6,:] =  -1.*(1. - N_spec/dp.Nh)*dp.Tt4 #dp.Tt4*(1. - N_spec/dp.Nh)
            #Res[6,:] = -1.0*(1. - N_spec/dp.Nh)#dp.Tt4 #dp.Tt4*(1. - N_spec/dp.Nh)


            #low pressure turbine
            deltah_lt =  -1./(1.+f)*1./dp.etam_lt*((ht2_5 - ht1_9)+ dp.aalpha*(ht2_1 - ht2))
            #deltah_lt =  -1./(1.+f)*1./dp.etam_lt*((ht2_5 - ht2_1)+ dp.aalpha*(ht2_1 - ht2))
            Tt4_9     = Tt4_5 + deltah_lt/Cp
            Pt4_9     = Pt4_5*(Tt4_9/Tt4_5)**(gamma/((gamma-1.)*dp.eta_lt))
            ht4_9     = ht4_5 + deltah_lt



            Res[7,:] = (dp.Pt5 - Pt4_9*dp.pi_tn)#/dp.Pt5


            #print f,dp.mhc,Tt4_1,Tt2_5,Pt2_5,Pt4_1,dp.mhtD


            results.Res = Res
            results.F   = F
            #results.sfc = TSFC


            ##----------compute analytic jacobian---------------------------------------------------------
            ##-----------------------------------------------------------------------------------
            ##dNfn_pif = 1.0
            ##dNln_pilc = 1.0
            ##dNfn_mf = 1.0
            ##dNln_mlc = 1.0

            #lvals = len(M0)

            #Jac = np.zeros([8,8,lvals])

            ##gamma, R, Cp, P0[iarr], T0[iarr], M0[iarr], throttle[iarr], dp.pi_lc[iarr], dp.mlc[iarr], dp.pi_hc[iarr], dp.mhc[iarr], dp.pi_f[iarr], dp.mf[iarr], dp.pi_d[iarr], dp.pi_fn[iarr], dp.eta_fn[iarr], dp.pi_lc[iarr], dp.Tt4[iarr], dp.eta_b[iarr], dp.htf[iarr], dp.pi_b[iarr], dp.Pref, dp.Tref, dp.etam_ht[iarr], dp.eta_ht[iarr], dp.etam_lt[iarr], dp.aalpha[iarr], dp.eta_lt[iarr], dp.pi_tn[iarr], dp.Pt5[iarr], dp.Tt4_spec[iarr], dp.Tt4_Tt2[iarr], dp.M2, dp.M2_5, dp.GF, dp.mhtD, dp.mltD, dp.A7, dp.A5, dp.N_spec[iarr], Nln[iarr], Nhn[iarr], Nfn[iarr], dp.eta_lc[iarr], dp.eta_hc[iarr], dp.eta_f[iarr], M7[iarr], P7[iarr], M5[iarr], P5[iarr], dNln_pilc[iarr], dNln_mlc[iarr], dNfn_pif[iarr], dNfn_mf[iarr])

            ##print gamma, R, Cp, P0, T0, M0, throttle
            ##print dp.pi_lc, dp.mlc, dp.pi_hc, dp.mhc, dp.pi_f, dp.mf, dp.pi_d, dp.pi_fn, dp.eta_fn, dp.pi_lc, dp.Tt4
            ##print dp.eta_b, dp.htf, dp.pi_b, dp.Pref, dp.Tref, dp.etam_ht, dp.eta_ht, dp.etam_lt, dp.aalpha, dp.eta_lt
            ##print dp.pi_tn, dp.Pt5, dp.Tt4_spec, dp.Tt4_Tt2, dp.M2, dp.M2_5, dp.GF, dp.mhtD, dp.mltD, dp.A7, dp.A5, dp.N_spec
            ##print Nln, Nhn, Nfn, dp.eta_lc, dp.eta_hc, dp.eta_f, M7, P7, M5, P5, dNln_pilc, dNln_mlc, dNfn_pif, dNfn_mf
            ##print M0.shape,dp.pi_lc.shape

            ##print M0,dp.pi_lc
            ##M0_shape = M0.shape()
            #Jac_loc = np.zeros(64)

            #an_jacobian = Turbofan_Jacobian()

            #for iarr in range(0,lvals):

                ##print M0[iarr],dp.pi_lc[iarr]
                ##print P0[iarr],T0[iarr]
                ##self.an_jacobian.Analytic_Jacobian(gamma, R, Cp, P0[iarr], T0[iarr], M0[iarr], throttle[iarr], dp_pilc[iarr], dp_mlc[iarr], dp_pi_hc[iarr], dp_mhc[iarr], dp_pi_f[iarr], dp_mf[iarr], dp_pi_d[iarr], dp_pi_fn[iarr], dp_eta_fn[iarr], dp_pi_lc[iarr], dp_Tt4[iarr], dp_eta_b[iarr], dp_htf[iarr], dp_pi_b[iarr], dp_Pref[iarr], dp_Tref[iarr], dp_etam_ht[iarr], dp_eta_ht[iarr], dp_etam_lt[iarr], dp_aalpha[iarr], dp_eta_lt[iarr], dp_pi_tn[iarr], dp_Pt5[iarr], dp_Tt4_spec[iarr], dp_Tt4_Tt2[iarr], dp_M2[iarr], dp_M2_5[iarr], dp_GF[iarr], dp_mhtD[iarr], dp_mltD[iarr], dp_A7[iarr], dp_A5[iarr], dp_N_spec[iarr], Nln[iarr], Nhn[iarr], Nfn[iarr], dp_eta_lc[iarr], dp_eta_hc[iarr], dp_eta_f[iarr], M7[iarr], P7[iarr], M5[iarr], P5[iarr], dNln_pilc[iarr], dNln_mlc[iarr], dNfn_pif[iarr], dNfn_mf[iarr])
                #an_jacobian.Analytic_Jacobian(gamma, R, Cp, P0[0][iarr], T0[0][iarr], M0[0][iarr], throttle[0][iarr], dp.pi_lc[0][iarr], dp.mlc[0][iarr], dp.pi_hc[0][iarr], dp.mhc[0][iarr], dp.pi_f[0][iarr], dp.mf[0][iarr], dp.pi_d[0][iarr], dp.pi_fn[0][iarr], dp.eta_fn[0][iarr], dp.pi_lc[0][iarr], dp.Tt4[0][iarr], dp.eta_b[0][iarr], dp.htf, dp.pi_b[0][iarr], dp.Pref, dp.Tref, dp.etam_ht[0][iarr], dp.eta_ht[0][iarr], dp.etam_lt[0][iarr], dp.aalpha[0][iarr], dp.eta_lt[0][iarr], dp.pi_tn[0][iarr], dp.Pt5[0][iarr], dp.Tt4_spec[0][iarr], dp.Tt4_Tt2[0][iarr], dp.M2, dp.M2_5, dp.GF, dp.mhtD[0][0], dp.mltD[0][0], dp.A7[0][0], dp.A5[0][0], dp.N_spec[0][iarr], Nln[0][iarr], Nhn[0][iarr], Nfn[0][iarr], dp.eta_lc[0][iarr], dp.eta_hc[0][iarr], dp.eta_f[0][iarr], M7[0][iarr], P7[0][iarr], M5[0][iarr], P5[0][iarr], dNln_pilc[0][iarr], dNln_mlc[0][iarr], dNfn_pif[0][iarr], dNfn_mf[0][iarr],Jac_loc)
                ##Jac[:,:,iarr] =self.an_jacobian.Jac
                #print Jac_loc

            ##print Jac




        results.Fsp  = Fsp
        results.Isp  = Isp
        results.sfc  = sfc
        results.mdot = mdot




        return results



    def size(self,mach_number,altitude,delta_isa = 0.):

        #Unpack components
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


        results = self.evaluate(conditions, 0)

        self.offdesign_params = deepcopy(self.design_params)




        ones_1col = np.ones([1,1])
        altitude      = ones_1col*0.0
        mach_number   = ones_1col*0.0
        throttle      = ones_1col*1.0

        #call the atmospheric model to get the conditions at the specified altitude
        atmosphere_sls = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        p,T,rho,a,mu = atmosphere_sls.compute_values(altitude,0.0)

        # setup conditions
        conditions_sls = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()



        # freestream conditions

        conditions_sls.freestream.altitude           = np.atleast_1d(altitude)
        conditions_sls.freestream.mach_number        = np.atleast_1d(mach_number)

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
        conditions_sls.propulsion.throttle           =  np.atleast_1d(throttle)

        state_sls = Data()
        state_sls.numerics = Data()
        state_sls.conditions = conditions_sls
        results_sls = self.offdesign(state_sls)



        self.sealevel_static_thrust = results_sls.F

        return results






    def evaluate_thrust(self,state):
        #results_offdesign = self.offdesign(state)

        #imports
        conditions = state.conditions
        numerics   = state.numerics
        #reference = self.reference
        throttle = conditions.propulsion.throttle

        local_state = copy.deepcopy(state)
        local_throttle = copy.deepcopy(throttle)

        #throttle = 0.6 + conditions.propulsion.throttle*(1.0-0.6)/1.0
        local_throttle[throttle<0.6] = 0.6
        local_throttle[throttle>1.0] = 1.0

        local_state.conditions.propulsion.throttle = local_throttle


        #freestream properties
        T0 = conditions.freestream.temperature
        p0 = conditions.freestream.pressure
        M0 = conditions.freestream.mach_number


        F = np.zeros([len(T0),3])
        mdot0 = np.zeros([len(T0),1])
        S  = np.zeros(len(T0))
        F_mdot0 = np.zeros(len(T0))

        results_eval = self.offdesign(local_state)

        local_scale = throttle/local_throttle


        F[:,0] = results_eval.F*local_scale.T
        mdot0[:,0] = results_eval.mdot*local_scale.T
        S = results_eval.TSFC

        #print throttle.T,local_throttle.T,F[:,0],local_scale.T





        ## setup conditions
        #conditions_eval = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()

        #state_eval = Data()
        #state_eval.numerics = Data()



        #for ieval in range(0,len(M0)):

            ## freestream conditions

            #conditions_eval.freestream.altitude           = np.atleast_1d(10.)
            #conditions_eval.freestream.mach_number        = np.atleast_1d(M0[ieval])

            #conditions_eval.freestream.pressure           = np.atleast_1d(p0[ieval][0])
            #conditions_eval.freestream.temperature        = np.atleast_1d(T0[ieval][0])
            ##conditions_eval.freestream.density            = np.atleast_1d(rho)
            ##conditions_eval.freestream.dynamic_viscosity  = np.atleast_1d(mu)
            ##conditions_eval.freestream.gravity            = np.atleast_1d(9.81)
            ##conditions_eval.freestream.gamma              = np.atleast_1d(1.4)
            ##conditions_eval.freestream.Cp                 = 1.4*287.87/(1.4-1)
            ##conditions_eval.freestream.R                  = 287.87
            ##conditions_eval.freestream.speed_of_sound     = np.atleast_1d(a)
            ##conditions_eval.freestream.velocity           = conditions_eval.freestream.mach_number * conditions_eval.freestream.speed_of_sound

            ## propulsion conditions
            #conditions_eval.propulsion.throttle           =  np.atleast_1d(throttle[ieval])


            #state_eval.conditions = conditions_eval
            #results_eval = self.offdesign(state_eval)

            #F[ieval][0] = results_eval.F
            #mdot0[ieval][0] = results_eval.mdot
            #S[ieval] = results_eval.TSFC
            ##F_mdot0[ieval] = results_eval.thrust_non_dim




        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot0
        results.sfc                 = S
        #results.thrust_non_dim      = F_mdot0
        #results.offdesigndata = results_eval.offdesigndata



        return results




    def offdesign(self,state):


        dp = deepcopy(self.design_params)
        self.offdesign_params = dp



        conditions = state.conditions
        throttle = conditions.propulsion.throttle

        lvals = len(throttle)

        self.init_vectors(conditions,dp)
        #print

        #print conditions.freestream.temperature,conditions.freestream.pressure

        results = self.evaluate(conditions,1)
        R = results.Res
        bp = self.set_baseline_params(len(throttle))
        #print "Residual : ",R,np.linalg.norm(R)
        #print "Results : ", results.F,results.sfc,self.offdesign_params.Nf,self.offdesign_params.Nl,self.offdesign_params.Nh,self.offdesign_params.mf/self.offdesign_params.mfD,self.offdesign_params.mlc/self.offdesign_params.mlcD,self.offdesign_params.mhc/self.offdesign_params.mhcD
        #print "Updates  : ",bp
        #print

        #print "Results : ", results.Fsp,results.F,results.sfc,self.offdesign_params.Nf,self.offdesign_params.Nl,self.offdesign_params.Nh,self.offdesign_params.mf/self.offdesign_params.mfD,self.offdesign_params.mlc/self.offdesign_params.mlcD,self.offdesign_params.mhc/self.offdesign_params.mhcD
        #print "Res : ",R


        d_bp = np.zeros([8,lvals])

        #print "T",throttle.T,results.F*self.number_of_engines

        if(np.linalg.norm(R)>1e-8):



            for iiter in range(0,self.max_iters):


                J = self.jacobian(conditions,bp,R)
                #print "Jac : ",J
                #print "Res : ",R

                #print J.shape,R.shape,d_bp.shape
                #print J[:,:,0].shape,R[:,0].shape,d_bp[:,0].shape

                for iarr in range(0,lvals):
                    d_bp[:,iarr] = -np.linalg.solve(J[:,:,iarr], R[:,iarr])
                    #d_bp[:,np.arange(0,lvals)] = -np.linalg.solve(J[:,:,np.arange(0,lvals)], R[:,np.arange(0,lvals)])


                #bp[6] = bp[6] + d_bp[6]


                ###compute the line step
                #d_a1 = 1.0
                ##d_a2 = d_a2/2.0
                #results = self.evaluate(conditions,1)
                #R = results.Res

                ##print bp,bp.shape
                #step_valid = 0
                #while (step_valid == 0):
                    #temp_dp = self.set_tem_baseline_params(self.offdesign_params,bp,d_a1,d_bp)
                    #results_1 = self.evaluate(conditions,1,temp_dp)
                    #R_1 = results_1.Res
                    #R1_nan = np.isnan(R_1)
                    ##print R1_nan
                    ##if (any(R1_nan.all())):
                        ##print any(R1_nan),R1_nan
                        ##d_a1 = 0.5*d_a1
                    ##else:
                        ##d_a2 = 0.5*d_a1
                        ##temp_dp2 = self.set_tem_baseline_params(self.offdesign_params,bp,d_a2,d_bp)
                        ##results_2 = self.evaluate(conditions,1,temp_dp2)
                        ##R_2 = results_2.Res
                        ##break

                    #if (R1_nan.all() == False):

                        #d_a2 = 0.5*d_a1
                        #temp_dp2 = self.set_tem_baseline_params(self.offdesign_params,bp,d_a2,d_bp)
                        #results_2 = self.evaluate(conditions,1,temp_dp2)
                        #R_2 = results_2.Res
                        #break
                    #else:
                        ##print R1_nan
                        ##print R1_nan.all is False
                        #d_a1 = 0.5*d_a1


                ##print d_a1,d_a2
                #x1 = bp
                #x2 = bp + d_a2*d_bp
                #x3 = bp + d_a1*d_bp

                #f_x1 = np.linalg.norm(R)
                #f_x2 = np.linalg.norm(R_2)
                #f_x3 = np.linalg.norm(R_1)

                #beta_23 = x2**2.0 - x3**2.0
                #beta_31 = x3**2.0 - x1**2.0
                #beta_12 = x1**2.0 - x2**2.0
                #gamma_23 = x2 - x3
                #gamma_31 = x3 - x1
                #gamma_12 = x1 - x2
                #d_eval = 0.5*(beta_23*f_x1 + beta_31*f_x2 + beta_12*f_x3)/(gamma_23*f_x1 + gamma_31*f_x2 + gamma_12*f_x3)


                #bp = d_eval #bp + self.newton_relaxation*d_bp



                bp = bp + self.newton_relaxation*d_bp
                self.update_baseline_params(bp)
                results = self.evaluate(conditions,1)
                R = results.Res

                #print R
                #print



                #R2 = np.copy(R)
                #R2[4] = 0.0

                #print "Residual : ",R,np.linalg.norm(R),np.linalg.norm(R2),self.offdesign_params.N_spec


                #print "Results : ", R.T , results.Fsp,results.F,results.sfc,self.offdesign_params.Nf,self.offdesign_params.Nl,self.offdesign_params.Nh,self.offdesign_params.mf/self.offdesign_params.mfD,self.offdesign_params.mlc/self.offdesign_params.mlcD,self.offdesign_params.mhc/self.offdesign_params.mhcD



                #print "Updates  : ",bp
                #print



                #print "T",throttle.T,results.F*self.number_of_engines


                if(np.linalg.norm(R)<1e-6):
                    break

                #if((R.all<1e-2) and (np.linalg.norm(R2)<1e-9)):
                    #break







        results_offdesign = Data()
        results_offdesign.F    = results.F*self.number_of_engines
        results_offdesign.TSFC = results.sfc
        results_offdesign.mdot = results.mdot*self.number_of_engines
        results_offdesign.Tt4  = dp.Tt4_spec
        results_offdesign.Tt3  = dp.Tt3
        results_offdesign.Pt3  = dp.Pt3
        results_offdesign.pi_lc = dp.pi_lc
        results_offdesign.pi_hc = dp.pi_hc
        results_offdesign.pi_f = dp.pi_f
        results_offdesign.flow = dp.mdot_core
        results_offdesign.aalpha = dp.aalpha
        results_offdesign.mdot_fan = dp.aalpha*dp.mdot_core
        results_offdesign.Nl = self.offdesign_params.Nl
        results_offdesign.Nf = self.offdesign_params.Nf
        results_offdesign.Nh = self.offdesign_params.Nh
        results_offdesign.mlc = self.offdesign_params.mlc
        results_offdesign.mhc = self.offdesign_params.mhc
        results_offdesign.mf = self.offdesign_params.mf

        #print results_offdesign.F,throttle.T,results_offdesign.TSFC

        return results_offdesign








    def init_vectors(self,conditions,dp):

        lvals = len(conditions.propulsion.throttle)
        onesv = np.ones([1,lvals])

        dp.aalpha  = dp.aalpha*onesv

        dp.pi_d    = dp.pi_d*onesv
        dp.eta_d   = dp.eta_d*onesv

        dp.pi_f    = dp.pi_f*onesv
        dp.eta_f   = dp.eta_f*onesv

        dp.pi_fn   = dp.pi_fn*onesv
        dp.eta_fn  = dp.eta_fn*onesv

        dp.pi_lc   = dp.pi_lc*onesv
        dp.eta_lc  = dp.eta_lc*onesv

        dp.pi_hc   = dp.pi_hc*onesv
        dp.eta_hc  = dp.eta_hc*onesv

        dp.Tt4     = dp.Tt4*onesv
        dp.pi_b    = dp.pi_b*onesv
        dp.eta_b   = dp.eta_b*onesv

        dp.eta_lt  = dp.eta_lt*onesv
        dp.etam_lt = dp.etam_lt*onesv

        dp.eta_ht  = dp.eta_ht*onesv
        dp.etam_ht = dp.etam_ht*onesv

        dp.pi_tn   = dp.pi_tn*onesv
        dp.eta_tn  = dp.eta_tn*onesv

        dp.mhc = dp.mhc*onesv
        dp.mlc = dp.mlc*onesv
        dp.mf  = dp.mf*onesv

        dp.Pt5  = dp.Pt5*onesv
        dp.Tt4_Tt2 = dp.Tt4_Tt2*onesv





    def set_tem_baseline_params(self,offdesign_params,bp,d_a1,d_bp):

        dp = copy.deepcopy(offdesign_params)
        #bp = np.zeros([8,lvals])
        #set the baseline params from the odp array
        dp.pi_f[0,:]  = bp[0,:] + d_a1*d_bp[0,:]
        dp.pi_lc[0,:] = bp[1,:] + d_a1*d_bp[1,:]
        dp.pi_hc[0,:] = bp[2,:] + d_a1*d_bp[2,:]
        dp.mf[0,:]    = bp[3,:] + d_a1*d_bp[3,:]
        dp.mlc[0,:]   = bp[4,:] + d_a1*d_bp[4,:]
        dp.mhc[0,:]   = bp[5,:] + d_a1*d_bp[5,:]
        dp.Tt4[0,:]   = bp[6,:] + d_a1*d_bp[6,:]
        dp.Pt5[0,:]   = bp[7,:] + d_a1*d_bp[7,:]

        return dp




    def set_baseline_params(self,lvals):

        dp = self.offdesign_params
        bp = np.zeros([8,lvals])
        #set the baseline params from the odp array
        bp[0,:] = dp.pi_f[0,:]
        bp[1,:] = dp.pi_lc[0,:]
        bp[2,:] = dp.pi_hc[0,:]
        bp[3,:] = dp.mf[0,:]
        bp[4,:] = dp.mlc[0,:]
        bp[5,:] = dp.mhc[0,:]
        bp[6,:] = dp.Tt4[0,:]
        bp[7,:] = dp.Pt5[0,:]

        return bp





    def update_baseline_params(self,bp):

        dp = self.offdesign_params
        #set the baseline params from the odp array
        dp.pi_f[0,:]  = bp[0,:]
        dp.pi_lc[0,:] = bp[1,:]
        dp.pi_hc[0,:] = bp[2,:]
        dp.mf[0,:]    = bp[3,:]
        dp.mlc[0,:]   = bp[4,:]
        dp.mhc[0,:]   = bp[5,:]
        dp.Tt4[0,:]   = bp[6,:]
        dp.Pt5[0,:]   = bp[7,:]


        return




    def jacobian(self,conditions,bp,R):
        dd = 1e-8
        dp_temp = deepcopy(self.offdesign_params)
        lvals = len(conditions.propulsion.throttle)
        #J = np.identity(8)
        J = np.zeros([8,8,lvals])

        dp_temp.pi_f[0,:]  = bp[0,:]*(1.+dd)
        results = self.evaluate(conditions, 1., dp_temp)
        dp_temp.pi_f[0,:] = bp[0,:]
        J[0,:,:] = (results.Res - R)/(bp[0,:]*dd)
        #J[:,0,:] = np.transpose((results.Res - R)/(bp[0,:]*dd))
        #print results.Res - R,bp[0]*dd,(results.Res - R)/(bp[0]*dd)



        dp_temp.pi_lc[0,:] = bp[1,:]*(1.+dd)
        results = self.evaluate(conditions, 1., dp_temp)
        dp_temp.pi_lc[0,:] = bp[1,:]
        J[1,:,:] = (results.Res - R)/(bp[1,:]*dd)
        #J[:,1,:] = np.transpose((results.Res - R)/(bp[1,:]*dd))


        dp_temp.pi_hc[0,:] = bp[2,:]*(1.+dd)
        results = self.evaluate(conditions, 1., dp_temp)
        dp_temp.pi_hc[0,:] = bp[2,:]
        J[2,:,:] = (results.Res - R)/(bp[2,:]*dd)
        #J[:,2,:] = np.transpose((results.Res - R)/(bp[2,:]*dd))


        dp_temp.mf[0,:]    = bp[3,:]*(1.+dd)
        results = self.evaluate(conditions, 1., dp_temp)
        dp_temp.mf[0,:] = bp[3,:]
        J[3,:,:] = (results.Res - R)/(bp[3,:]*dd)
        #J[:,3,:] = np.transpose((results.Res - R)/(bp[3,:]*dd))


        dp_temp.mlc[0,:]   = bp[4,:]*(1.+dd)
        results = self.evaluate(conditions, 1., dp_temp)
        dp_temp.mlc[0,:] = bp[4,:]
        J[4,:,:] = (results.Res - R)/(bp[4,:]*dd)
        #J[:,4,:] = np.transpose((results.Res - R)/(bp[4,:]*dd))


        dp_temp.mhc[0,:]   = bp[5,:]*(1.+dd)
        results = self.evaluate(conditions, 1., dp_temp)
        dp_temp.mhc[0,:] = bp[5,:]
        J[5,:,:] = (results.Res - R)/(bp[5,:]*dd)
        #J[:,5,:] = np.transpose((results.Res - R)/(bp[5,:]*dd))


        dp_temp.Tt4[0,:]   = bp[6,:]*(1.+dd)
        results = self.evaluate(conditions, 1., dp_temp)
        dp_temp.Tt4[0,:] = bp[6,:]
        J[6,:,:] = (results.Res - R)/(bp[6,:]*dd)
        #J[:,6,:] = np.transpose((results.Res - R)/(bp[6,:]*dd))


        dp_temp.Pt5[0,:]   = bp[7,:]*(1.+dd)
        results = self.evaluate(conditions, 1., dp_temp)
        dp_temp.Pt5[0,:] = bp[7,:]
        J[7,:,:] = (results.Res - R)/(bp[7,:]*dd)
        #J[:,7,:] = np.transpose((results.Res - R)/(bp[7,:]*dd))

        J = np.swapaxes(J, 0, 1)
        #J = J.T

        #print J
        return J





    def performance(self,pid,mdot,flag):

        dp = self.design_params
        odp = self.offdesign_params

        #dp.mhc = mhcD
        #dp.mlc = mlcD
        #dp.mf  = mfD

        eng_paras = Data()


        #low pressure compressure
        if(flag == 0):
            #pid = pid/dp.pi_lc #pid*dp.pi_hc/dp.pi_lc
            #mdot = mdot*odp.mhcD/odp.mlcD
            pb = (pid-1.)/(dp.pi_lc-1.)
            #mb = mdot/odp.mhcD

            #pb = (pid-1.)/(dp.pi_lc-1.)
            mb                = mdot/odp.mlcD #mdot*odp.mhcD/odp.mlcD #mdot/odp.mlcD
            eng_paras.etapol0 = dp.eta_lc
            eng_paras.mb0     = odp.mlcD #odp.mhcD #odp.mlcD
            #eng_paras.a       = 1.5
            #eng_paras.da      = 0.5
            #eng_paras.c       = 3.
            #eng_paras.d       = 4.
            #eng_paras.CC      = 15.0
            #eng_paras.DD      = 1.0
            #eng_paras.k       = 0.03
            #eng_paras.b       = 5.0
            #eng_paras.Nd      = odp.Nln
            eng_paras.CC2     = 0.1

            eng_paras.a       = 3.0
            eng_paras.da      = 0.5
            eng_paras.c       = 3.0
            eng_paras.d       = 6.0
            eng_paras.CC      = 2.5
            eng_paras.DD      = 15.0
            eng_paras.k       = 0.03
            eng_paras.b       = 0.85
            eng_paras.Nd      = odp.Nln
            eng_paras.piD = dp.pi_lc

            #pb2 = pid/dp.pi_lc*1.795
            #mb2 = mb





            #pid = pid*26.0/dp.pi_lc
            ##mdot = mdot*odp.mhcD/odp.mlcD
            #pb = (pid-1.)/(26.0-1.)
            ##mb = mdot/odp.mhcD

            ##mb = mdot*odp.mhcD/odp.mlcD #mdot/odp.mlcD
            #mb = mdot/odp.mlcD #mdot/odp.mlcD
            #eng_paras.etapol0 = dp.eta_lc
            #eng_paras.mb0     = odp.mhcD #odp.mlcD
            #eng_paras.a       = 1.5
            #eng_paras.da      = 0.5
            #eng_paras.c       = 3.
            #eng_paras.d       = 4.
            #eng_paras.CC      = 15.0
            #eng_paras.DD      = 1.0
            #eng_paras.k       = 0.03
            #eng_paras.b       = 5.0






        #high pressure compressor
        elif(flag == 1):
            pb = (pid-1.)/(dp.pi_hc-1.)
            mb = mdot/odp.mhcD
            eng_paras.etapol0 = dp.eta_hc
            eng_paras.mb0     = odp.mhcD
            eng_paras.a       = 1.5
            eng_paras.da      = 0.5
            eng_paras.c       = 3.
            eng_paras.d       = 4.
            eng_paras.CC      = 15.0
            eng_paras.DD      = 1.0
            eng_paras.k       = 0.03
            eng_paras.b       = 5.0
            eng_paras.Nd      = odp.Nhn
            eng_paras.CC2     = 0.1
            eng_paras.piD = dp.pi_hc

            #pb2 = pid/dp.pi_hc*1.795
            #mb2 = mb


            #pid = pid*26.0/dp.pi_hc

            #pb = (pid-1.)/(26.0-1.)
            ##mb = mdot/odp.mhcD
            #mb = mdot/odp.mhcD
            #eng_paras.etapol0 = dp.eta_hc
            #eng_paras.mb0     = odp.mhcD
            #eng_paras.a       = 1.5
            #eng_paras.da      = 0.5
            #eng_paras.c       = 3.
            #eng_paras.d       = 4.
            #eng_paras.CC      = 15.0
            #eng_paras.DD      = 1.0
            #eng_paras.k       = 0.03
            #eng_paras.b       = 5.0


        #fan
        elif(flag == 2):
            pb = (pid-1.)/(dp.pi_f-1.)
            mb = mdot/odp.mfD
            eng_paras.etapol0 = dp.eta_f
            eng_paras.mb0     = odp.mfD
            eng_paras.a       = 3.0
            eng_paras.da      = -0.5
            eng_paras.c       = 3.0
            eng_paras.d       = 6.0
            eng_paras.CC      = 2.5
            eng_paras.DD      = 15.0
            eng_paras.k       = 0.03
            eng_paras.b       = 0.85
            eng_paras.Nd      = odp.Nfn
            eng_paras.CC2     = 0.1
            eng_paras.piD = dp.pi_f

            #pb2 = pid/dp.pi_f*1.795
            #mb2 = mb



        #inputs = np.zeros(2)
        #inputs[0] = mb2
        #inputs[1] = pb2

        Nb,dN_dpi,dN_dm  = self.speed_map(flag,pb,mb,eng_paras)
        etapol = self.efficiency_map(flag,pb,mb,eng_paras)
        #etapol  = self.evaluate_compressor_map_gpr(inputs)

        #print Nb
        return Nb,etapol,dN_dpi,dN_dm






    def speed_map(self,flag,pb,mb,eng_paras):
        odp  = self.offdesign_params
        dp = self.design_params

        #compute non dim params dp.Nfc,dp.Nlc,dp.Nhc
        etapol0 = eng_paras.etapol0
        mb0     = eng_paras.mb0
        a       = eng_paras.a
        da      = eng_paras.da
        c       = eng_paras.c
        d       = eng_paras.d
        CC      = eng_paras.CC
        DD      = eng_paras.DD
        b       = eng_paras.b
        k       = eng_paras.k
        Nd      = eng_paras.Nd
        piD     = eng_paras.piD


        R = 1.0
        Nb = 0.5*np.ones(pb.shape)#/eng_paras.Nd
        dN = 1.e-8

        while (np.linalg.norm(R)>1e-8):

            ms = Nb**b
            ps = ms**a

            Nb_1 = Nb*(1.+dN)
            ms_1 = Nb_1**b
            ps_1 = ms_1**a

            R  = ms + k*(1. - np.exp((pb - ps)/(2.*Nb*k))) - mb
            R1 = ms_1 + k*(1. - np.exp((pb - ps_1)/(2.*Nb_1*k))) - mb
            dR = (R1-R)/(dN*Nb)

            #if(pb>=mb**a):


                #R  = ms + k*(1. - np.exp((pb - ps)/(2.*Nb*k))) - mb
                #R1 = ms_1 + k*(1. - np.exp((pb - ps_1)/(2.*Nb_1*k))) - mb
                #dR = (R1-R)/(dN*Nb)
                #vvv = 'a'

            #else:

                #R  = ms + k*(1. - np.exp((pb - ps)/(2.*Nb*k))) - mb
                #R1 = ms_1 + k*(1. - np.exp((pb - ps_1)/(2.*Nb_1*k))) - mb
                #dR = (R1-R)/(dN*Nb)
                #vvv = 'b'


            delN = -R/dR #-R/dR
            Nb += delN
            NNNN = 1


        #print R,pb,mb**a,vvv
        grad_n = 0.5/Nb*np.exp((pb - ps)/(2.*Nb*k))
        grad_d = b*Nb**(b-1.0) + 0.5/Nb**2.0*np.exp((pb - ps)/(2.*Nb*k))*(a*b*Nb**(a*b) + pb - ps)

        dN_dpi = Nd*(grad_n/grad_d)/(piD-1.0)
        dN_dm  = Nd/(mb0*grad_d)



        Nb = Nb*Nd

        #compute gradient




        return Nb,dN_dpi,dN_dm







    def efficiency_map(self,flag,pb,mb,eng_paras):
        odp  = self.offdesign_params
        dp = self.design_params

        #compute efficiencies dp.eta_lc,dp.eta_hc,dp.eta_f

        etapol0 = eng_paras.etapol0
        mb0     = eng_paras.mb0
        a       = eng_paras.a
        da      = eng_paras.da
        c       = eng_paras.c
        d       = eng_paras.d
        CC      = eng_paras.CC
        DD      = eng_paras.DD
        b       = eng_paras.b
        k       = eng_paras.k
        CC2     = eng_paras.CC2


        #etapol = etapol0*(1. - CC*(np.abs(pb/mb**(a+da-1.) - mb))**c - DD*(np.abs(mb/mb0-1.))**d)
        #etapol = etapol0*(1. - CC*(np.abs(pb/mb**(a+da-1.) - mb))**c )
        #etapol = etapol0*(1. - (np.abs(pb-1.0))**c - (np.abs(mb-1.0))**d)
        etapol = etapol0*(1. - CC2*(np.abs(pb/mb-1.0))**c)


        #print pb/mb**(a+da-1.),mb, (np.abs(pb-1.0))**c #(np.abs(pb/mb**(a+da-1.)-mb))**c

        #print etapol

        return etapol



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

        #return



    __call__ = evaluate_thrust





def Tt(M,T,gamma):
    return T*(1.+((gamma-1.)/2. *M**2.))

def Pt(M,P,gamma):
    return P*((1.+(gamma-1.)/2. *M**2. )**3.5)


def Tt_inv(M,Tt,gamma):
    return Tt/(1.+((gamma-1.)/2. *M**2.))


def Pt_inv(M,Pt,gamma):
    return Pt/((1.+(gamma-1.)/2. *M**2. )**3.5)

