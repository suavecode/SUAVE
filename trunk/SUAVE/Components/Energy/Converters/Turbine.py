## @ingroup Components-Energy-Converters
# Turbine.py
#
# Created:  Jul 2014, A. Variyar
# Modified: Aug 2016, L. Kulik

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

# package imports
import numpy as np
import scipy as sp

from SUAVE.Components.Energy.Energy_Component import Energy_Component


# ----------------------------------------------------------------------
#  Turbine Component
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
class Turbine(Energy_Component):
    """This is a turbine component typically used in a turbofan.
    Calling this class calls the compute function.
    
    Assumptions:
    Efficiencies do not change with varying conditions.

    Source:
    https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/
    """
    
    def __defaults__(self):
        """ This sets the default values for the component to function.

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
        #set the default values
        self.tag ='Turbine'
        self.mechanical_efficiency             = 1.0
        self.polytropic_efficiency             = 1.0
        self.inputs.stagnation_temperature     = 1.0
        self.inputs.stagnation_pressure        = 1.0
        self.inputs.fuel_to_air_ratio          = 1.0
        self.outputs.stagnation_temperature    = 1.0
        self.outputs.stagnation_pressure       = 1.0
        self.outputs.stagnation_enthalpy       = 1.0

        self.inputs.shaft_power_off_take       = None
    
    
    def compute(self,conditions):
        """This computes the output values from the input values according to
        equations from the source.

        Assumptions:
        Constant polytropic efficiency and pressure ratio

        Source:
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/

        Inputs:
        conditions.freestream.
          isentropic_expansion_factor         [-]
          specific_heat_at_constant_pressure  [J/(kg K)]
        self.inputs.
          stagnation_temperature              [K]
          stagnation_pressure                 [Pa]
          bypass_ratio                        [-]
          fuel_to_air_ratio                   [-]
          compressor.work_done                [J/kg]
          fan.work_done                       [J/kg]
          shaft_power_off_take.work_done      [J/kg]

        Outputs:
        self.outputs.
          stagnation_temperature              [K]  
          stagnation_pressure                 [Pa]
          stagnation_enthalpy                 [J/kg]

        Properties Used:
        self.
          mechanical_efficiency               [-]
          polytropic_efficiency               [-]
        """           
        #unpack the values
        
        #unpack from conditions
        gamma           = conditions.freestream.isentropic_expansion_factor
        Cp              = conditions.freestream.specific_heat_at_constant_pressure
        
        #unpack from inputs
        Tt_in           = self.inputs.stagnation_temperature
        Pt_in           = self.inputs.stagnation_pressure
        alpha           = self.inputs.bypass_ratio
        f               = self.inputs.fuel_to_air_ratio
        compressor_work = self.inputs.compressor.work_done
        fan_work        = self.inputs.fan.work_done

        if self.inputs.shaft_power_off_take is not None:
            shaft_takeoff = self.inputs.shaft_power_off_take.work_done
        else:
            shaft_takeoff = 0.

        #unpack from self
        eta_mech        =  self.mechanical_efficiency
        etapolt         =  self.polytropic_efficiency
        
        #method to compute turbine properties
        
        #Using the work done by the compressors/fan and the fuel to air ratio to compute the energy drop across the turbine
        deltah_ht = -1 / (1 + f) * 1 / eta_mech * (compressor_work + shaft_takeoff + alpha * fan_work)
        
        #Compute the output stagnation quantities from the inputs and the energy drop computed above
        Tt_out    =  Tt_in+deltah_ht/Cp
        Pt_out    =  Pt_in*(Tt_out/Tt_in)**(gamma/((gamma-1)*etapolt))
        ht_out    =  Cp*Tt_out   #h(Tt4_5)
        
        
        #pack the computed values into outputs
        self.outputs.stagnation_temperature  = Tt_out
        self.outputs.stagnation_pressure     = Pt_out
        self.outputs.stagnation_enthalpy     = ht_out
    
    
    __call__ = compute