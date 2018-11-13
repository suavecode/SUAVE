## @ingroup Sizing
#size_from_PGM.py

# Created : Jun 2016, M. Vegh
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
import numpy as np

from SUAVE.Core import Data, Units
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import fuselage_planform
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Propulsion import compute_turbofan_geometry
from SUAVE.Methods.Geometry.Two_Dimensional.Planform.rescale_non_dimensional import set_origin_dimensional
from SUAVE.Methods.Propulsion import turbofan_sizing
from SUAVE.Methods.Propulsion import turbojet_sizing

# ----------------------------------------------------------------------
#  Size from PGM
# ----------------------------------------------------------------------

def size_from_PGM(vehicle):
        """Completes the sizing of a SUAVE vehicle to determine fill out all of the dimensions of the vehicle.
           This takes in the vehicle as it is provided from the PGM analysis
    
            Assumptions:
            Simple tapered wing (no sections)
            Simple fuselage  (no sections)
    
            Source:
            N/A
    
            Inputs:
            vehicle    [SUAVE Vehicle]
    
            Outputs:
            vehicle    [SUAVE Vehicle]
    
            Properties Used:
            None
        """        
        
        # The top level info
        vehicle.systems.control     = "fully powered" 
        vehicle.systems.accessories = "medium range"        
        vehicle.envelope.ultimate_load = 2.5
        vehicle.envelope.limit_load    = 1.5
        vehicle.mass_properties.takeoff = vehicle.mass_properties.max_takeoff
        
        ####################################
        vehicle.mass_properties.max_zero_fuel
        ####################################
        
        # Passengers
        vehicle.passengers  = vehicle.performance.vector[0][-1] *1.
        
        for fuse in vehicle.fuselages:
                fuse.number_coach_seats = vehicle.passengers 
                fuse.differential_pressure = 55. * 1000.* Units.pascals
                
        
        # Size the wings
        max_area = 0
        for wing in vehicle.wings:
                
                # Use existing scripts
                wing = wing_planform(wing)
                
                # Get the max area
                if wing.areas.reference>max_area:
                        max_area = wing.areas.reference
        
        # Size the fuselage
        for fuse in vehicle.fuselages:
                
                 # Use existing scripts
                fuse = fuselage_planform(fuse)
                fuse.heights.at_quarter_length              = fuse.heights.maximum    
                fuse.heights.at_three_quarters_length       = fuse.heights.maximum    
                fuse.heights.at_wing_root_quarter_chord     = fuse.heights.maximum    
                fuse.heights.at_vertical_root_quarter_chord = fuse.heights.maximum    
        
        # Size the propulsion system
        for prop in vehicle.propulsors:
                prop.number_of_engines = int(np.round(prop.number_of_engines))
                
                orig = prop.origin[0]
                prop.origin.clear()
                for eng in range(prop.number_of_engines):
                        prop.origin.append(orig)
                if prop.tag == 'Turbofan':
                        
                        turbofan = prop
                        
                        conditions = None
                        
                        # ------------------------------------------------------------------
                        #   Component 1 - Ram
                
                        # to convert freestream static to stagnation quantities
                
                        # instantiate
                        ram = SUAVE.Components.Energy.Converters.Ram()
                        ram.tag = 'ram'
                
                        # add to the network
                        turbofan.append(ram)
                
                
                        # ------------------------------------------------------------------
                        #  Component 2 - Inlet Nozzle
                
                        # instantiate
                        inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
                        inlet_nozzle.tag = 'inlet_nozzle'
                
                        # setup
                        inlet_nozzle.polytropic_efficiency = 0.98
                        inlet_nozzle.pressure_ratio        = 0.98
                
                        # add to network
                        turbofan.append(inlet_nozzle)
                
                
                        # ------------------------------------------------------------------
                        #  Component 3 - Low Pressure Compressor
                
                        # instantiate 
                        compressor = SUAVE.Components.Energy.Converters.Compressor()    
                        compressor.tag = 'low_pressure_compressor'
                
                        # setup
                        compressor.polytropic_efficiency = 0.91
                        compressor.pressure_ratio        = 1.14    
                
                        # add to network
                        turbofan.append(compressor)
                
                
                        # ------------------------------------------------------------------
                        #  Component 4 - High Pressure Compressor
                
                        # instantiate
                        compressor = SUAVE.Components.Energy.Converters.Compressor()    
                        compressor.tag = 'high_pressure_compressor'
                
                        # setup
                        compressor.polytropic_efficiency = 0.91
                        compressor.pressure_ratio        = 13.415    
                
                        # add to network
                        turbofan.append(compressor)
                
                
                        # ------------------------------------------------------------------
                        #  Component 5 - Low Pressure Turbine
                
                        # instantiate
                        turbine = SUAVE.Components.Energy.Converters.Turbine()   
                        turbine.tag='low_pressure_turbine'
                
                        # setup
                        turbine.mechanical_efficiency = 0.99
                        turbine.polytropic_efficiency = 0.93     
                
                        # add to network
                        turbofan.append(turbine)
                
                
                        # ------------------------------------------------------------------
                        #  Component 6 - High Pressure Turbine
                
                        # instantiate
                        turbine = SUAVE.Components.Energy.Converters.Turbine()   
                        turbine.tag='high_pressure_turbine'
                
                        # setup
                        turbine.mechanical_efficiency = 0.99
                        turbine.polytropic_efficiency = 0.93     
                
                        # add to network
                        turbofan.append(turbine)
                
                
                        # ------------------------------------------------------------------
                        #  Component 7 - Combustor
                
                        # instantiate    
                        combustor = SUAVE.Components.Energy.Converters.Combustor()   
                        combustor.tag = 'combustor'
                
                        # setup
                        combustor.efficiency                = 0.99 
                        combustor.alphac                    = 1.0     
                        combustor.turbine_inlet_temperature = 1450
                        combustor.pressure_ratio            = 0.95
                        combustor.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()    
                
                        # add to network
                        turbofan.append(combustor)
                
                
                        # ------------------------------------------------------------------
                        #  Component 8 - Core Nozzle
                
                        # instantiate
                        nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
                        nozzle.tag = 'core_nozzle'
                
                        # setup
                        nozzle.polytropic_efficiency = 0.95
                        nozzle.pressure_ratio        = 0.99    
                
                        # add to network
                        turbofan.append(nozzle)
                
                
                        # ------------------------------------------------------------------
                        #  Component 9 - Fan Nozzle
                
                        # instantiate
                        nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
                        nozzle.tag = 'fan_nozzle'
                
                        # setup
                        nozzle.polytropic_efficiency = 0.95
                        nozzle.pressure_ratio        = 0.99    
                
                        # add to network
                        turbofan.append(nozzle)
                
                
                        # ------------------------------------------------------------------
                        #  Component 10 - Fan
                
                        # instantiate
                        fan = SUAVE.Components.Energy.Converters.Fan()   
                        fan.tag = 'fan'
                
                        # setup
                        fan.polytropic_efficiency = 0.93
                        fan.pressure_ratio        = 1.7    
                
                        # add to network
                        turbofan.append(fan)
                
                
                        # ------------------------------------------------------------------
                        #Component 10 : thrust (to compute the thrust)
                        thrust = SUAVE.Components.Energy.Processes.Thrust()       
                        thrust.tag ='compute_thrust'
                
                        #total design thrust (includes all the engines)
                        thrust.total_design  = turbofan.number_of_engines *  turbofan.sealevel_static_thrust
                
                        # add to network
                        turbofan.thrust = thrust                       
                        
                        prop.working_fluid = SUAVE.Attributes.Gases.Air()
                        prop = compute_turbofan_geometry(prop, conditions)
                        turbofan_sizing(prop,mach_number = 0.1, altitude = 0., delta_isa = 0)
                        
                if prop.tag == 'Turbojet':
                        
                        
                        
                        # ------------------------------------------------------------------
                        #   Component 1 - Ram
                        
                        # to convert freestream static to stagnation quantities
                        
                        # instantiate
                        ram = SUAVE.Components.Energy.Converters.Ram()
                        ram.tag = 'ram'
                        
                        # add to the network
                        prop.append(ram)
                    
                    
                        # ------------------------------------------------------------------
                        #  Component 2 - Inlet Nozzle
                        
                        # instantiate
                        inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
                        inlet_nozzle.tag = 'inlet_nozzle'
                        
                        # setup
                        inlet_nozzle.polytropic_efficiency = 1.0
                        inlet_nozzle.pressure_ratio        = 1.0
                        inlet_nozzle.pressure_recovery     = 0.94
                        
                        # add to network
                        prop.append(inlet_nozzle)
                        
                        
                        # ------------------------------------------------------------------
                        #  Component 3 - Low Pressure Compressor
                        
                        # instantiate 
                        compressor = SUAVE.Components.Energy.Converters.Compressor()    
                        compressor.tag = 'low_pressure_compressor'
                    
                        # setup
                        compressor.polytropic_efficiency = 0.88
                        compressor.pressure_ratio        = 3.1    
                        
                        # add to network
                        prop.append(compressor)
                    
                        
                        # ------------------------------------------------------------------
                        #  Component 4 - High Pressure Compressor
                        
                        # instantiate
                        compressor = SUAVE.Components.Energy.Converters.Compressor()    
                        compressor.tag = 'high_pressure_compressor'
                        
                        # setup
                        compressor.polytropic_efficiency = 0.88
                        compressor.pressure_ratio        = 5.0  
                        
                        # add to network
                        prop.append(compressor)
                    
                    
                        # ------------------------------------------------------------------
                        #  Component 5 - Low Pressure Turbine
                        
                        # instantiate
                        turbine = SUAVE.Components.Energy.Converters.Turbine()   
                        turbine.tag='low_pressure_turbine'
                        
                        # setup
                        turbine.mechanical_efficiency = 0.99
                        turbine.polytropic_efficiency = 0.89
                        
                        # add to network
                        prop.append(turbine)
                        
                          
                        # ------------------------------------------------------------------
                        #  Component 6 - High Pressure Turbine
                        
                        # instantiate
                        turbine = SUAVE.Components.Energy.Converters.Turbine()   
                        turbine.tag='high_pressure_turbine'
                    
                        # setup
                        turbine.mechanical_efficiency = 0.99
                        turbine.polytropic_efficiency = 0.87
                        
                        # add to network
                        prop.append(turbine)
                          
                        
                        # ------------------------------------------------------------------
                        #  Component 7 - Combustor
                        
                        # instantiate    
                        combustor = SUAVE.Components.Energy.Converters.Combustor()   
                        combustor.tag = 'combustor'
                        
                        # setup
                        combustor.efficiency                = 0.94
                        combustor.alphac                    = 1.0     
                        combustor.turbine_inlet_temperature = 1440.
                        combustor.pressure_ratio            = 0.92
                        combustor.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()    
                        
                        # add to network
                        prop.append(combustor)
                        
                        # ------------------------------------------------------------------
                        #  Afterburner
                        
                        # instantiate    
                        afterburner = SUAVE.Components.Energy.Converters.Combustor()   
                        afterburner.tag = 'afterburner'
                        
                        # setup
                        afterburner.efficiency                = 0.9
                        afterburner.alphac                    = 1.0     
                        afterburner.turbine_inlet_temperature = 1500
                        afterburner.pressure_ratio            = 1.0
                        afterburner.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()    
                        
                        # add to network
                        prop.append(afterburner)    
                    
                        
                        # ------------------------------------------------------------------
                        #  Component 8 - Core Nozzle
                        
                        # instantiate
                        nozzle = SUAVE.Components.Energy.Converters.Supersonic_Nozzle()   
                        nozzle.tag = 'core_nozzle'
                        
                        # setup
                        nozzle.pressure_recovery     = 0.95
                        nozzle.pressure_ratio        = 1.   
                        
                        # add to network
                        prop.append(nozzle)
                        
                        
                        # ------------------------------------------------------------------
                        #Component 10 : thrust (to compute the thrust)
                        thrust = SUAVE.Components.Energy.Processes.Thrust()       
                        thrust.tag ='compute_thrust'
                        
                        #total design thrust (includes all the engines)
                        thrust.total_design = prop.number_of_engines * prop.sealevel_static_thrust
                     
                        # Note: Sizing builds the propulsor. It does not actually set the size of the turbojet
                        #design sizing conditions
                        altitude      = 0.0*Units.ft
                        mach_number   = 0.2
                        isa_deviation = 0.    
                        
                        # add to network
                        prop.thrust = thrust
                        
                        prop.working_fluid = SUAVE.Attributes.Gases.Air()
                    
                        #size the turbojet                  
                        turbojet_sizing(prop,mach_number = 0.1, altitude = 0., delta_isa = 0)          
                        conditions = None
                        prop = compute_turbofan_geometry(prop, conditions)                   

        # Vehicle reference area
        try:
                area = vehicle.wings.main_wing.areas.reference
        except:
                area = max_area
        
        vehicle.reference_area = area    
    
        # Set the origins
        vehicle = set_origin_dimensional(vehicle)
    
    
        return vehicle