# Thrust.py
#
# Created:  Jul 2014, A. Variyar
# Modified: Feb 2016, T. MacDonald, A. Variyar, M. Vegh


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

from SUAVE.Core import Units

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp


from SUAVE.Core import Data
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Components.Propulsors.Propulsor import Propulsor


# ----------------------------------------------------------------------
#  Thrust Process
# ----------------------------------------------------------------------

class Thrust_JDM(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Thrust
        a component that computes the thrust and other output properties

        this class is callable, see self.__call__

        """

    def __defaults__(self):

        #setting the default values
        self.tag ='Thrust'
        self.bypass_ratio                             = 0.0
        self.compressor_nondimensional_massflow       = 0.0
        self.reference_temperature                    = 288.15
        self.reference_pressure                       = 1.01325*10**5
        self.number_of_engines                        = 0.0
        self.inputs.fuel_to_air_ratio                 = 0.0  #changed
        self.outputs.thrust                           = 0.0
        self.outputs.thrust_specific_fuel_consumption = 0.0
        self.outputs.specific_impulse                 = 0.0
        self.outputs.non_dimensional_thrust           = 0.0
        self.outputs.core_mass_flow_rate              = 0.0
        self.outputs.fuel_flow_rate                   = 0.0
        self.outputs.fuel_mass                        = 0.0 #changed
        self.outputs.power                            = 0.0
        self.design_thrust                            = 0.0
        self.mass_flow_rate_design                    = 0.0



    def compute(self,conditions):

        #unpack the values
        aalpha  = self.inputs.aalpha
        a0      = self.inputs.a0
        g_c     = 1.0
        f       = self.inputs.f
        V9_a0   = self.inputs.V9_a0
        M0      = conditions.freestream.mach_number
        R_t     = self.inputs.R_t
        T9_T0   = self.inputs.T9_T0
        P0_P9   = self.inputs.P0_P9
        R_c     = self.inputs.R_c
        gamma_c = self.inputs.gamma_c
        V19_a0  = self.inputs.V19_a0
        T19_T0  = self.inputs.T19_T0
        P0_P19  = self.inputs.P0_P19

        F_mdot0 = 1.0/(1.0+aalpha)*a0/g_c*( (1.0+f)*V9_a0 - M0 + (1.0+f)*R_t*T9_T0*(1-P0_P9)/(R_c*V9_a0*gamma_c)) + aalpha/(1.0 + aalpha)*a0/g_c*(V19_a0 - M0 + T19_T0*(1-P0_P19)/(V19_a0*gamma_c))

        S = f/((1.0+aalpha)*F_mdot0)*3600.0*2.20462/0.224809

        #pack outputs

        self.outputs.mass_specific_thrust       = F_mdot0
        self.outputs.specific_fuel_consumption  = S




    __call__ = compute

