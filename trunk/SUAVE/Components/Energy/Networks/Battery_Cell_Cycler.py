## @ingroup Components-Energy-Networks
# Battery_Cell_Cycler.py
# 
# Created: Apr 2021, M. Clarke
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from .Network import Network  
from SUAVE.Methods.Power.Battery.Cell_Cycle_Models.LiNCA_cell_cycle_model      import compute_NCA_cell_state_variables 
from SUAVE.Methods.Power.Battery.Cell_Cycle_Models.LiNiMnCoO2_cell_cycle_model import compute_NMC_cell_state_variables
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Battery_Cell_Cycler(Network):
    """ This is a test bench to analyze the discharge and charge profile 
        of a battery cell.
    
        Assumptions:
        None
        
        Source:
        None
    """  
    def __defaults__(self):
        """ This sets the default values for the network to function.
    
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
        self.avionics                = None
        self.voltage                 = None  
        
    # manage process with a driver function
    def evaluate_thrust(self,state):
        """ Evaluate the state variables of a cycled cell
    
            Assumptions: 
            None
    
            Source:
            N/A
    
            Inputs:
            state [state()]
    
            Outputs: 
            conditions.propulsion: 
                current                       [amps]
                battery_power_draw            [watts]
                battery_energy                [joules]
                battery_voltage_open_circuit  [volts]
                battery_voltage_under_load    [volts]  
    
            Properties Used:
            Defaulted values
        """          
    
        # unpack
        conditions        = state.conditions
        numerics          = state.numerics 
        avionics          = self.avionics 
        battery           = self.battery    
         
        # Set battery energy
        battery.current_energy      = conditions.propulsion.battery_energy
        battery.pack_temperature    = conditions.propulsion.battery_pack_temperature
        battery.charge_throughput   = conditions.propulsion.battery_charge_throughput     
        battery.age_in_days         = conditions.propulsion.battery_age_in_days 
        battery_discharge_flag      = conditions.propulsion.battery_discharge_flag    
        battery.R_growth_factor     = conditions.propulsion.battery_resistance_growth_factor
        battery.E_growth_factor     = conditions.propulsion.battery_capacity_fade_factor 
        battery.max_energy          = conditions.propulsion.battery_max_aged_energy 
        n_series                    = battery.pack_config.series  
        n_parallel                  = battery.pack_config.parallel 
        n_total                     = n_series*n_parallel
    
        # update ambient temperature based on altitude
        battery.ambient_temperature                   = conditions.freestream.temperature   
        battery.ambient_pressure                      = conditions.freestream.pressure
        battery.cooling_fluid.thermal_conductivity    = conditions.freestream.thermal_conductivity
        battery.cooling_fluid.kinematic_viscosity     = conditions.freestream.kinematic_viscosity
        battery.cooling_fluid.density                 = conditions.freestream.density 
          
        # Predict voltage based on battery  
        volts = battery.compute_voltage(state)
 
        #-------------------------------------------------------------------------------
        # Discharge
        #-------------------------------------------------------------------------------
        if battery_discharge_flag:
            # Calculate avionics and payload power
            avionics_power = np.ones((numerics.number_control_points,1))*avionics.current * volts
        
            # Calculate avionics and payload current
            avionics_current =  np.ones((numerics.number_control_points,1))*avionics.current    
            
            # link
            battery.inputs.current  = avionics_current
            battery.inputs.power_in = -avionics_power
            battery.inputs.voltage  = volts
            battery.energy_calc(numerics,battery_discharge_flag)          
            
        else: 
            battery.inputs.current  = -battery.charging_current * np.ones_like(volts)
            battery.inputs.voltage  =  battery.charging_voltage * np.ones_like(volts) 
            battery.inputs.power_in =  -battery.inputs.current * battery.inputs.voltage * np.ones_like(volts)
            battery.energy_calc(numerics,battery_discharge_flag)        
        
        # Pack the conditions for outputs     
        conditions.propulsion.battery_thevenin_voltage             = battery.thevenin_voltage 
        conditions.propulsion.battery_current                      = abs( battery.current )
        conditions.propulsion.battery_power_draw                   = battery.inputs.power_in 
        conditions.propulsion.battery_energy                       = battery.current_energy  
        conditions.propulsion.battery_charge_throughput            = battery.charge_throughput 
        conditions.propulsion.battery_state_of_charge              = battery.state_of_charge
        conditions.propulsion.battery_voltage_open_circuit         = battery.voltage_open_circuit
        conditions.propulsion.battery_voltage_under_load           = battery.voltage_under_load  
        conditions.propulsion.battery_internal_resistance          = battery.internal_resistance
        conditions.propulsion.battery_age_in_days                  = battery.age_in_days
        conditions.propulsion.battery_pack_temperature             = battery.pack_temperature 
        conditions.propulsion.battery_max_aged_energy              = battery.max_energy  
        conditions.propulsion.battery_charge_throughput            = battery.cell_charge_throughput    
        conditions.propulsion.battery_specfic_power                = -battery.inputs.power_in/battery.mass_properties.mass   
        
        conditions.propulsion.battery_cell_power_draw              = -conditions.propulsion.battery_power_draw/n_series
        conditions.propulsion.battery_cell_energy                  = battery.current_energy/n_total   
        conditions.propulsion.battery_cell_voltage_under_load      = battery.cell_voltage_under_load    
        conditions.propulsion.battery_cell_voltage_open_circuit    = battery.cell_voltage_open_circuit  
        conditions.propulsion.battery_cell_current                 = abs(battery.cell_current)         
        conditions.propulsion.battery_cell_heat_energy_generated   = battery.heat_energy_generated    
        conditions.propulsion.battery_cell_temperature             = battery.cell_temperature
        conditions.propulsion.battery_cell_joule_heat_fraction     = battery.cell_joule_heat_fraction   
        conditions.propulsion.battery_cell_entropy_heat_fraction   = battery.cell_entropy_heat_fraction   
            
          
        F     = np.zeros_like(volts)  * [0,0,0]      
        mdot  = state.ones_row(1)*0.0
         
        results                     = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot  
        
        return results 
    

     
    def unpack_unknowns(self,segment):
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            unknowns specific to the battery cell 
    
            Outputs: 
            conditions specific to the battery cell
             
            Properties Used:
            N/A
        """             
        # unpack 
        battery = self.battery 
        
        # append battery unknowns 
        battery.append_battery_unknowns(segment)   
        return  

    
    
    def residuals(self,segment):
        """ This packs the residuals to be sent to the mission solver.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            unknowns specific to the battery cell 
            
            Outputs:
            residuals specific to the battery cell
    
            Properties Used: 
            N/A
        """         
        # unpack 
        network = self
        battery = self.battery 
        
        # append battery residuals 
        battery.append_battery_residuals(segment,network)   
                
        return     

    def add_unknowns_and_residuals_to_segment(self, segment, initial_voltage = None, 
                                              initial_battery_cell_temperature = 300 , initial_battery_state_of_charge = 0.5,
                                              initial_battery_cell_current = 5. ,initial_battery_cell_thevenin_voltage= 0.1 ):
        """ This function sets up the information that the mission needs to run a mission segment using this network
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:  
            initial_voltage                       [volts]
            initial_battery_cell_temperature      [Kelvin]
            initial_battery_state_of_charge       [unitless]
            initial_battery_cell_current          [Amperes]
            initial_battery_cell_thevenin_voltage [Volts]
            
            Outputs
            None
            
            Properties Used:
            N/A
        """         
        # unpack  
        battery  = self.battery 
        
        segment.state.residuals.network  = Data()      
        
        # add unknowns and residuals specific to battery cell
        battery.append_battery_unknowns_and_residuals_to_segment(segment,initial_voltage,
                                              initial_battery_cell_temperature , initial_battery_state_of_charge,
                                              initial_battery_cell_current,initial_battery_cell_thevenin_voltage)  
        
        # Ensure the mission knows how to pack and unpack the unknowns and residuals
        segment.process.iterate.unknowns.network  = self.unpack_unknowns
        segment.process.iterate.residuals.network = self.residuals        

        return segment
    
    __call__ = evaluate_thrust


