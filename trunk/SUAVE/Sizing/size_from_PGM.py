## @ingroup Sizing
#size_from_PGM.py

# Created : Jun 2016, M. Vegh
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
import numpy as np
import scipy as sp

from SUAVE.Core import Data, Units
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_segmented_planform
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import fuselage_planform
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Propulsion import compute_turbofan_geometry
from SUAVE.Methods.Geometry.Two_Dimensional.Planform.rescale_non_dimensional import set_origin_dimensional
from SUAVE.Methods.Propulsion import turbofan_sizing
from SUAVE.Methods.Propulsion import turbojet_sizing

from SUAVE.Components.Wings.Main_Wing import Main_Wing, Segment_Container
from SUAVE.Components.Wings.Segment import Segment

# ----------------------------------------------------------------------
#  Size from PGM
# ----------------------------------------------------------------------


def loc(x,t_c,height):
        
        x[x<0] = 0.
        
        the_height = 2.*5.*t_c*(0.2969*x**0.5-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
        
        val = the_height - height
        
        return val

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
        vehicle.envelope.ultimate_load = 3.75
        vehicle.envelope.limit_load    = 2.5
        vehicle.mass_properties.takeoff = vehicle.mass_properties.max_takeoff
        
        # Deal with passengers
        pax = 0
        n_fuses = len(vehicle.fuselages)
        
        # Size the wings
        max_area = 0
        for wing in vehicle.wings:
                
                # Use existing scripts
                if len(wing.Segments)>0:
                        
                        # Need to fix segments here. Make sure there's a tip and a root and order them sequentially
                        wing = fix_wing_segments(wing)
                        
                        # Size the wing
                        wing = wing_segmented_planform(wing)
                        
                else:
                        wing = wing_planform(wing)
          
                # Get the max area
                if isinstance(wing,Main_Wing):
                        max_area += wing.areas.reference
                        
                if n_fuses == 0.:
                        
                        # This is a flying wing. Figure out how many pax can fit
                        
                        # The height needs to be at least 1.4 meters
                        height = 1.4
                        
                        # Pull out data
                        t_c   = wing.thickness_to_chord
                        c     = wing.chords.root   
                        span  = wing.spans.projected
                        taper = wing.taper
                        
                        # Determine the area which has the min height
                        h_c = height/c
                        max_height = t_c*c
                        
                        if height<max_height:
                        
                                root = lambda x:loc(x,t_c,h_c)
                                
                                x0 = np.array([0.])
                                x1 = np.array([1.])
                                
                                fore_location = sp.optimize.newton(root, x0)
                                aft_location  = sp.optimize.newton(root, x1)
                                
                                root_length = (aft_location - fore_location)*c
                                
                                # Assume the passengers are in the inner 10% for now
                                per        = 0.1
                                span_loc   = per*span
                                end_chord  = c*((1-per)-per*taper)
                                max_height = end_chord*t_c
                                
                                if height<max_height:
                                        
                                        x0 = np.array([0.])
                                        x1 = np.array([1.])                                        
                                        
                                        root = lambda x:loc(x,t_c,h_c)
                                        
                                        fore_location = sp.optimize.newton(root, x0)
                                        aft_location  = sp.optimize.newton(root, x1)
                                        
                                        end_length = (aft_location - fore_location)*end_chord                      
                                        
                                        avg_length = 0.5*(root_length+end_length)
                                        seats_len  = avg_length / (31.* Units.inches)
                                        
                                        # Calculate seats abreast
                                        bins  = np.floor(span_loc/0.53)
                                        if bins<=7:
                                                aisles = 1.
                                        elif bins>7:
                                                aisles = 2.
                                        elif bins>=15:
                                                aisles = 3.
                                        elif bins>= 23:
                                                aisles = 4.
                                                
                                        seats_abreast = bins-aisles
                                        
                                        seats = np.floor(seats_abreast*seats_len)
                                        
                                        if np.isnan(seats):
                                                seats = 0.
                                        elif seats<= 0.:
                                                seats = 0.
                                                
                                        pax  += seats
                                        
                                        vehicle.passengers = int(pax)
                                        
                                        if vehicle.passengers <=0:
                                                print('neg pax wing')

        # Set the vehicle reference area
        vehicle.reference_area = max_area   
                            
        # Size the fuselage
        
       
        for fuse in vehicle.fuselages:
                
                # Split the number of passengers over the vehicle
                #fuse.number_coach_seats = vehicle.passengers/n_fuses
                fuse.differential_pressure = 55. * 1000.* Units.pascals 
                fuse.seat_pitch            = 31. * Units.inch
                
                # If we've got a concorde
                if vehicle.performance.vector[0][2]>344.:
                        fuse.seat_pitch            = 38. * Units.inch
                        
                
                # Calculate seats abreast
                bins  = np.floor(fuse.width/0.53)
                if bins<=7:
                        aisles = 1.
                elif bins>7:
                        aisles = 2.
                fuse.seats_abreast = bins-aisles
                
                 # Use existing scripts
                fuse = fuselage_planform(fuse)
                fuse.heights.at_quarter_length              = fuse.heights.maximum    
                fuse.heights.at_three_quarters_length       = fuse.heights.maximum    
                fuse.heights.at_wing_root_quarter_chord     = fuse.heights.maximum    
                fuse.heights.at_vertical_root_quarter_chord = fuse.heights.maximum
                
                pax += fuse.number_coach_seats
                
                vehicle.passengers = pax
                
                if vehicle.passengers <0:
                        print('neg pax fuse')                
                
                fuse.OpenVSP_values = Data() # VSP uses degrees directly
        
                fuse.OpenVSP_values.nose = Data()
                fuse.OpenVSP_values.nose.top = Data()
                fuse.OpenVSP_values.nose.side = Data()
                fuse.OpenVSP_values.nose.top.angle = 75.0
                fuse.OpenVSP_values.nose.top.strength = 0.40
                fuse.OpenVSP_values.nose.side.angle = 45.0
                fuse.OpenVSP_values.nose.side.strength = 0.75  
                fuse.OpenVSP_values.nose.TB_Sym = True
                fuse.OpenVSP_values.nose.z_pos = -.015
        
                fuse.OpenVSP_values.tail = Data()
                fuse.OpenVSP_values.tail.top = Data()
                fuse.OpenVSP_values.tail.side = Data()    
                fuse.OpenVSP_values.tail.bottom = Data()
                fuse.OpenVSP_values.tail.top.angle = -90.0
                fuse.OpenVSP_values.tail.top.strength = 0.1
                fuse.OpenVSP_values.tail.side.angle = -30.0
                fuse.OpenVSP_values.tail.side.strength = 0.50  
                fuse.OpenVSP_values.tail.TB_Sym = True
                fuse.OpenVSP_values.tail.bottom.angle = -30.0
                fuse.OpenVSP_values.tail.bottom.strength = 0.50 
                fuse.OpenVSP_values.tail.z_pos = .035                
        
        # Size the propulsion system
        for prop in vehicle.propulsors:
                prop.number_of_engines = int(np.round(prop.number_of_engines))
                
                #prop.OpenVSP_flow_through = True
                
                orig = prop.origin[0]
                prop.origin = []
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
                        turbofan_sizing(prop,mach_number = 0.01, altitude = 0., delta_isa = 0)
                        
                        if prop.engine_length == 0.:
                                prop.engine_length = 0.01
                                
                        if prop.nacelle_diameter == 0.:
                                prop.nacelle_diameter= 0.01    
                                
                        #prop.OpenVSP_flow_through = True
                        
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
                        turbojet_sizing(prop,mach_number = 0.01, altitude = 0., delta_isa = 0)          
                        conditions = None
                        prop = compute_turbofan_geometry(prop, conditions)            
                        
                        if prop.engine_length == 0.:
                                prop.engine_length = 0.01
                                
                        if prop.nacelle_diameter == 0.:
                                prop.nacelle_diameter= 0.01                           

        # Set the origins
        vehicle = set_origin_dimensional(vehicle)
        
    
        return vehicle


def fix_wing_segments(wing):
        
        # Unpack:
        segs = wing.Segments
                
        
        # Reorder the segments: make a new container.
        new_container = Segment_Container()
        
        # Loop over and pull out the locations
        spanwise_locs = []
        for key,seg in segs.items():
                spanwise_locs.append(seg.percent_span_location)
        
        # The indices that would sort the segs
        indices = np.argsort(spanwise_locs)

        # Extrapolate to root and tip, if not at 0 and 1
        if spanwise_locs[indices[0]]>=1E-3:
                # Need a root
                seg_0 = Segment()
                
                # See if Extrapolation is even possible:
                if len(segs)==1:
                
                        seg_0.twist = segs[indices[0]].twist
                        seg_0.root_chord_percent = segs[indices[0]].root_chord_percent
                        seg_0.dihedral_outboard = segs[indices[0]].dihedral_outboard 
                        seg_0.sweeps.quarter_chord = segs[indices[0]].sweeps.quarter_chord
                        seg_0.thickness_to_chord = segs[indices[0]].thickness_to_chord
                else:
                        delta_span = spanwise_locs[indices[1]]-spanwise_locs[indices[0]]
                        
                        # Extrapolate linearly: twist, root chord percent, dihedral, sweeps.quarter_chord, thickness_to_chord
                        seg_0.twist = segs[indices[0]].twist - segs[indices[0]].percent_span_location*(segs[indices[1]].twist-segs[indices[0]].twist)/delta_span
                        #seg_0.root_chord_percent = segs[indices[0]].root_chord_percent - \
                                #segs[indices[0]].percent_span_location*(segs[indices[1]].root_chord_percent-segs[indices[0]].root_chord_percent)/delta_span
                        seg_0.root_chord_percent = 1.0
                                
                        seg_0.dihedral_outboard = segs[indices[0]].dihedral_outboard - \
                                segs[indices[0]].percent_span_location*(segs[indices[1]].dihedral_outboard-segs[indices[0]].dihedral_outboard)/delta_span
                        seg_0.sweeps.quarter_chord = segs[indices[0]].sweeps.quarter_chord -\
                                segs[indices[0]].percent_span_location*(segs[indices[1]].sweeps.quarter_chord-segs[indices[0]].sweeps.quarter_chord)/delta_span
                        seg_0.thickness_to_chord = segs[indices[0]].thickness_to_chord - \
                                segs[indices[0]].percent_span_location*(segs[indices[1]].thickness_to_chord-segs[indices[0]].thickness_to_chord)/delta_span
                        
                        # Check for things less than 0
                        if seg_0.root_chord_percent<0:
                                seg_0.root_chord_percent = 0.
                        if seg_0.thickness_to_chord<0.0001:
                                seg_0.thickness_to_chord = 0.0001
                
                new_container.append(seg_0)
                
        elif spanwise_locs[indices[0]]<1E-3:
                segs[indices[0]].percent_span_location = 0.
                segs[indices[0]].root_chord_percent    = 1.0
                
                
                
        for ind in indices:
                if len(new_container) !=0:
                        if new_container[-1].percent_span_location != segs[ind].percent_span_location:
                                new_container.append(segs[ind])
                else:
                        new_container.append(segs[ind])
                
        if spanwise_locs[indices[-1]]<(1.-1E-3):
                
                # May need a tip
                seg_m1 = Segment()
                
                # See if Extrapolation is even possible:
                if len(segs)==1:
                        
                        seg_m1.percent_span_location = 1.0
                        seg_m1.twist = segs[indices[0]].twist
                        seg_m1.root_chord_percent = segs[indices[0]].root_chord_percent
                        seg_m1.dihedral_outboard = segs[indices[0]].dihedral_outboard 
                        seg_m1.sweeps.quarter_chord = segs[indices[0]].sweeps.quarter_chord
                        seg_m1.thickness_to_chord = segs[indices[0]].thickness_to_chord
                else:                

                        delta_span = spanwise_locs[indices[-1]]-spanwise_locs[indices[-2]]
                        
                        seg_m1.percent_span_location = 1.0
                        seg_m1.twist = segs[indices[-1]].twist + (1-segs[indices[-1]].percent_span_location)*(segs[indices[-1]].twist-segs[indices[-2]].twist)/delta_span
                        seg_m1.root_chord_percent = segs[indices[-1]].root_chord_percent + (1-segs[indices[-1]].percent_span_location)*(segs[indices[-1]].root_chord_percent-\
                                                                                         segs[indices[-2]].root_chord_percent)/delta_span
                        seg_m1.dihedral_outboard = segs[indices[-1]].dihedral_outboard + (1-segs[indices[-1]].percent_span_location)*(segs[indices[-1]].dihedral_outboard-\
                                                                                         segs[indices[-2]].dihedral_outboard)/delta_span
                        seg_m1.sweeps.quarter_chord = segs[indices[-1]].sweeps.quarter_chord + (1-segs[indices[-1]].percent_span_location)*(segs[indices[-1]].sweeps.quarter_chord-\
                                                                                         segs[indices[-2]].sweeps.quarter_chord)/delta_span
                        seg_m1.thickness_to_chord = segs[indices[-1]].thickness_to_chord + (1-segs[indices[-1]].percent_span_location)*(segs[indices[-1]].thickness_to_chord-\
                                                                                         segs[indices[-2]].thickness_to_chord)/delta_span
                        
                        # Check for things less than 0
                        if seg_m1.root_chord_percent<0:
                                seg_m1.root_chord_percent = 0.
                        if seg_m1.thickness_to_chord<0.0001:
                                seg_m1.thickness_to_chord = 0.0001
                                
                new_container.append(seg_m1)
                                
        if spanwise_locs[indices[-1]]>=(1.-1E-3):
                segs[indices[-1]].percent_span_location = 1.
                
                                        
                

                
        # Check to ensure nothing exceeds the bounds
        min_bounds = np.array(segs[0].PGM_char_min_bounds)
        max_bounds = np.array(segs[0].PGM_char_max_bounds)
        keys       = segs[0].PGM_characteristics
        for seg in new_container:
                for ii, key in enumerate(keys):
                        val = new_container[seg].deep_get(key)
                        
                        if np.isnan(val): val = 0.; new_container[seg].deep_set(key,0.)
                        if val < min_bounds[ii]: new_container[seg].deep_set(key,min_bounds[ii])
                        if val > max_bounds[ii]: new_container[seg].deep_set(key,max_bounds[ii])
                        

        # Change the container out
        wing.Segments = new_container
        
        return wing