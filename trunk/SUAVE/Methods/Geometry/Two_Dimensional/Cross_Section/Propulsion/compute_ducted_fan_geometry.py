# engine_geometry.py
#
# Created:  Jun 15, A. Variyar 
# Modified: Mar 16, M. Vegh
# Modified: Aug 16, D. Bianchi
# Modified: Sep 16, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE
from SUAVE.Core  import Data

# package imports
import numpy as np
from math import pi, sqrt

# ----------------------------------------------------------------------
#  Correlation-based methods to compute engine geometry
# ----------------------------------------------------------------------

def compute_ducted_fan_geometry(ducted_fan, mach_number = None, altitude = None, delta_isa = 0, conditions = None):
    """ SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Propulsion.compute_ducted_fan_geometry
    inputs:
            ducted_fan: component ducted fan
            conditions: SUAVE structured data for sizing conditions (optional)
            alternatively the user can provide unstrucutred sizing conditions given by:
                mach_number: free stream mach number
                altitude: flight altitude
                delta_isa: temperature deviation from ISA conditions
    outputs:
            ducted fan geometry data:
                areas.wetted
                areas.maximum
                nacelle_diameter
                engine_length
    """

    

    #Unpack conditions

    #check if altitude is passed or conditions is passed

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
            atmo_data = atmosphere.compute_values(altitude,delta_isa)

            p   = atmo_data.pressure          
            T   = atmo_data.temperature       
            rho = atmo_data.density          
            a   = atmo_data.speed_of_sound    
            mu  = atmo_data.dynamic_viscosity  
            
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

    # unpack
    thrust            = ducted_fan.thrust
    fan_nozzle        = ducted_fan.fan_nozzle
    mass_flow         = thrust.mass_flow_rate_design

    #evaluate engine at these conditions
    state=Data()
    state.conditions=conditions
    state.numerics= Data()
    ducted_fan.evaluate_thrust(state)
    
    #determine geometry
    U0       = conditions.freestream.velocity
    rho0     = conditions.freestream.density
    Ue       = fan_nozzle.outputs.velocity
    rhoe     = fan_nozzle.outputs.density
    Ae       = mass_flow[0][0]/(rhoe[0][0]*Ue[0][0]) #ducted fan nozzle exit area
    A0       = (mass_flow/(rho0*U0))[0][0]
    

   
    ducted_fan.areas.maximum = 1.2*Ae/fan_nozzle.outputs.area_ratio[0][0]
    ducted_fan.nacelle_diameter = 2.1*((ducted_fan.areas.maximum/np.pi)**.5)

    ducted_fan.engine_length    = 1.5*ducted_fan.nacelle_diameter
    ducted_fan.areas.wetted     = ducted_fan.nacelle_diameter*ducted_fan.engine_length*np.pi

    
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print