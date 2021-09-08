## @ingroup Components-Energy-Processes
# Rocket_Thrust.py
#
# Created:  Feb 2018, W. Maier
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Rocket Thrust Process
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Processes
class Rocket_Thrust(Energy_Component):
    """A class that handles computation of thrust and ISP for rocket engines

    Assumptions:
    
    Source:
    Chapter 7
    https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/
    """         

    def __defaults__(self):
        """This sets the default value.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """                
        self.tag ='Thrust'
        self.number_of_engines                        = 0.0
        self.mass_flow_rate                           = 0.0
        self.inputs.expansion_ratio                   = 1.0
        self.inputs.area_throat                       = 1.0
        self.outputs.thrust                           = 0.0 
        self.outputs.specific_impulse                 = 0.0
        self.outputs.non_dimensional_thrust           = 0.0
        self.outputs.mass_flow_rate                   = 0.0
        self.design_thrust                            = 0.0
        self.ISP_design                               = 0.0

    def compute(self,conditions):
        """Computes thrust and other properties as below.

        Assumptions:
        Perfect gas

        Source:
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/

        Inputs:
        conditions.freestream.
          pressure                           [Pa]
          temperature                        [K]
          gravity                            [-]
        conditions.throttle                  [-]
        self.inputs.
          propellant_mass_rate               [kg/s]
          combustion_pressure                [Pa]
          expansion_ratio                    [-]
          area_throat                        [m^2]
          exhaust_velocity                   [m/s]
          number_of_engines                  [-]

        Outputs:
        self.outputs.
          thrust                             [N]
          propellant_mass_flow_rate          [kg/s]
          power                              [W]
          specific_impulse                   [s]
          exhuast_velocity                   [m/s]

        Properties Used:
        
        """  
        
        #--Unpack the values--

        # unpacking from conditions
        p0               = conditions.freestream.pressure  
        g0               = conditions.freestream.gravity
        throttle         = conditions.propulsion.throttle
        
        # unpacking from inputs
        Pt_combustion    = self.inputs.combustion_pressure
        pe               = self.inputs.static_pressure         
        expansion_ratio  = self.inputs.expansion_ratio
        area_throat      = self.inputs.area_throat
        Me               = self.inputs.mach_number
        exhaust_velocity = self.inputs.exhaust_velocity
        num_eng          = self.inputs.number_of_engines 
        gamma            = self.inputs.isentropic_expansion_factor
        rho              = self.inputs.density
              
        #--Computing the thrust coefficient and effective exhaust velocity--
        gmm = gamma*Me*Me
        C  = exhaust_velocity*(1.+1./(gmm)*(1.-p0/pe))
        
        #--Computing specific impulse
        Isp           = C/g0        
        
        #--Computing the propellant/exhaust mass rate
        Ae            = area_throat*expansion_ratio
        mdot_temp     = throttle*rho*exhaust_velocity*Ae
        
        #--Computing Dimensional Thrust
        thrust        = num_eng*mdot_temp*C
        
        #--Make mdot size of thrust
        mdot               = mdot_temp 
        mdot[mdot<0.0]     = 0.0 
        thrust[thrust<0.0] = 0.0
        
        #--Pack outputs--
        self.outputs.thrust                            = thrust  
        self.outputs.vehicle_mass_rate                 = mdot
        self.outputs.specific_impulse                  = Isp

    def size(self,conditions):
        """Sizes the engine for the design condition.

        Assumptions:
        Perfect gas

        Source:
        Chapter 7
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/

        Inputs:
        conditions.freestream.speed_of_sound [m/s] (conditions is also passed to self.compute(..))
        self.inputs.
          bypass_ratio                       [-]
          total_temperature_reference        [K]
          total_pressure_reference           [Pa]
          number_of_engines                  [-]

        Outputs:
        self.outputs.non_dimensional_thrust  [-]

        Properties Used:
        self.
          reference_temperature              [K]
          reference_pressure                 [Pa]
          total_design                       [N] - Design thrust
        """             
        # unpack inputs
        throttle      = 1.0
        g0            = conditions.freestream.gravity
        
        # unpack from self
        design_thrust = self.total_design
        design_ISP    = self.ISP_design
        num_eng       = self.inputs.number_of_engines

        # compute nondimensional thrust
        self.compute(conditions)
        
        # compute flow rate
        mdot          = design_thrust/(design_ISP*g0)*num_eng
        
        #--Pack outputs--
        self.mass_flow_rate = mdot

        return

    __call__ = compute         