""" Turbojet.py: Turbo_Jet 1D gasdynamic engine model """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure import Data
from Propulsor import Propulsor
from Segments import Inlet, Compressor, Burner, Turbine, Nozzle

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Turbojet(Propulsor):

    """ A Turbojet cycle Propulsor with optional afterburning """

    def __defaults__(self):
        
        # global
        self.tag = 'Turbo_Jet'
        self.power = Data()
        self.power.fuel_ox_ratio_max = 0.0      # kg/s
        self.power.P_mech = 0.0                 # W
        self.power.P_e = 0.0                    # W

        # thermodynamic properties
        self.thermo = Data()
        self.N = 9                              # number of stations 
        self.thermo.ht = np.zeros(self.N)            # J/kg-s
        self.thermo.Tt = np.zeros(self.N)            # K
        self.thermo.pt = np.zeros(self.N)            # Pa
        self.thermo.cp = np.zeros(self.N)            # J/kg-K
        self.thermo.gamma = np.zeros(self.N)         # cp/cv
        self.thermo.R = np.zeros(self.N)             # J/kg-K

        # engine segments
        self.Inlet = Inlet()
        self.Inlet.i = 0; self.Inlet.f = 1

        self.Compressors = Data()
        self.Compressors.LowPressure = Compressor()
        self.Compressors.LowPressure.i = 1 
        self.Compressors.LowPressure.f = 2

        self.Compressors.HighPressure = Compressor()
        self.Compressors.HighPressure.i = 2 
        self.Compressors.HighPressure.f = 3

        self.Burner = Burner()
        self.Burner.i = 3; self.Burner.f = 4

        self.Turbines = Data()
        self.Turbines.HighPressure = Turbine()
        self.Turbines.HighPressure.i = 4 
        self.Turbines.HighPressure.f = 5

        self.Turbines.LowPressure = Turbine()
        self.Turbines.LowPressure.i = 5 
        self.Turbines.LowPressure.f = 6

        self.Afterburner = Burner()
        self.Afterburner.i = 6; self.Afterburner.f = 7
        self.Afterburner.active = False

        self.Nozzle = Nozzle()
        self.Nozzle.i = 7; self.Nozzle.f = 8

    def initialize(self,F_max,atmosphere,alt_design=0.0,M_design=0.0,p_design='ideal'):

        # free stream conditions
        p_inf, T_inf, rho_inf, a_inf, mew_inf = atmosphere.compute_values(alt_design)
        cp_inf = self.gas.compute_cp(T_inf,p_inf)
        Tt_inf = gasdynamics.Tt(M_design,T_inf,g)
        ht_inf = Tt_inf*cp_inf
        pt_inf = gasdynamics.pt(M_design,p_inf,g)
        
        
        self.pt[0] = pt_inf
        self.Tt[0] = Tt_inf
        self.ht[0] = ht_inf

        # step through stages, find ht, Tt, and pt
        self.Inlet(self.thermo,self.power)
        self.Compressors.LowPressure(self.thermo,self.power)
        self.Compressors.HighPressure(self.thermo,self.power)
        self.Burner(self.thermo,self.power)
        self.Turbines.HighPressure(self.thermo,self.power)
        self.Turbines.LowPressure(self.thermo,self.power)
        self.Afterburner(self.thermo,self.power)
        self.Nozzle(self.thermo,self.power)

        # find exit conditions
        if p_design == 'ideal':
            p_e = p_inf
        else:
            if p_design is not None:
                if p_design > 0:
                    p_e = p_design
                else:
                    print "Error in engine initialization: design exit pressure must be >= 0."
            else:
                print "Error in engine initialization: design exit pressure not defined."


        Me = gasdynamics.M_from_Pt(self.Nozzle.pt[1],self.pe)
        Te = gasdynamics.T_from_Tt(self.Nozzle.Tt[1],Me)
        ae = np.sqrt(self.Nozzle.gamma[1]*self.Gas.R*Te)
        ue = Me*ae
        self.Ae = 0.0

        # find maximum thrust coeff
        self.mdot_intake = (F_max)/()

        CF_design = 0.0

        return CF_design

    def __call__(self,eta,segment):

        """  CF, Isp, Psp = evaluate(eta,segment): sum the performance of all the propulsion systems
    
            Inputs:    eta = array of throttle values                          (required)  (floats)    
                    segment                                                 (required)  (Mission Segment instance)

            Outputs:   CF = array of thurst coefficient values                 (floats)
                    Isp = array of specific impulse values (F/(dm/dt))      (floats)
                    Psp = array of specific power values (P/F)              (floats)                                                                                            

        """

        return
