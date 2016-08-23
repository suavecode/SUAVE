# print_engine_data.py

# Created: SUAVE team
# Updated: Carlos Ilario, Feb 2016

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np
import SUAVE
from SUAVE.Core import Units,Data

import time                     # importing library
import datetime                 # importing library
# ----------------------------------------------------------------------
#  Print output file with compressibility drag
# ----------------------------------------------------------------------
def print_engine_data(vehicle,filename = 'engine_data.dat'):
    """ SUAVE.Methods.Results.print_compress_drag(vehicle,filename = 'compress_drag.dat'):
        
        Print output file with compressibility drag
        
        Inputs:
            vehicle         - SUave type vehicle
            filename [optional] - Name of the file to be created

        Outputs:
            output file

        Assumptions:

    """ 
    
    
    d_isa_vec   = [0 , 10]
    speed_vec   = np.linspace( 0,   450, 10) * Units.knots
    speed_vec[0] = 1.
    hp_vec      = np.linspace( 0, 50000, 11) * Units.ft
    thrust      = np.zeros_like(speed_vec)
    mdot        = np.zeros_like(speed_vec)
    
    # Determining vehicle number of engines
    engine_number = 0.
    for propulsor in vehicle.propulsors : # may have than one propulsor
        engine_number += propulsor.number_of_engines
    if engine_number == 0:
        raise ValueError, "No engine found in the vehicle"

    #        
    engine_tag          = vehicle.propulsors.turbofan.tag
    design_thrust       = vehicle.propulsors.turbofan.design_thrust
    engine_length       = vehicle.propulsors.turbofan.engine_length
    nacelle_diameter    = vehicle.propulsors.turbofan.nacelle_diameter
    bypass_ratio        = vehicle.propulsors.turbofan.thrust.bypass_ratio
            
    # Considering planet and atmosphere of 1st mission segment
    sea_level_gravity   =  9.81 #mission.segments[0].planet.sea_level_gravity
    atmo                =  SUAVE.Analyses.Atmospheric.US_Standard_1976() #mission.segments[0].atmosphere
    
    p0, T0, rho0, a0, mu0 = atmo.compute_values(0,0)
    
    state = Data()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics() 
    state.numerics   = SUAVE.Analyses.Mission.Segments.Conditions.Numerics()
    
    # write header of file
    fid = open(filename,'w')   # Open output file
    fid.write('Output file with engine data \n\n') 
    fid.write('  VEHICLE TAG : ' + vehicle.tag + '\n\n')
  
   
    for d_isa in d_isa_vec:
        fid.write('\n DELTA ISA: {:3.0f} degC'.format(d_isa) + '\n')
        fid.write('\n' + 8*' ' + '|' + 40*' ' + ' SPEED [KTAS]' + 48*' ' + '|' + 27*' ' + ' SPEED [KTAS]' + 33*' ' )
        fid.write('\n' + 8*' ' + '|')
        fid.write(np.transpose(map('{:10.0f}'.format,speed_vec/Units.knots)))
        fid.write( ' |' + 3* ' ')        
        fid.write( np.transpose(map('{:6.0f} '.format,speed_vec/Units.knots)))            
        fid.write('\n HP[ft] |' + 40*' ' + ' THRUST [lbf]' + 48*' ' + '|' + 30*' ' + ' SFC [adm]'    + 33*' ' + '|\n')        
        
        for altitude in hp_vec:
            for idx,speed in enumerate(speed_vec):

                # Computing atmospheric conditions                
                p , T , rho , a , mu  = atmo.compute_values(altitude,0)
                T_delta_ISA = T + d_isa
                sigma_disa = (p/p0) / (T_delta_ISA/T0)
                rho_delta_ISA = rho0 * sigma_disa
                a_delta_ISA = atmo.fluid_properties.compute_speed_of_sound(T_delta_ISA)
                
                # Getting engine thrust
                state.conditions.freestream.dynamic_pressure = np.array(np.atleast_1d(0.5 * rho_delta_ISA * speed*speed))
                state.conditions.freestream.gravity          = np.array(np.atleast_1d(sea_level_gravity))
                state.conditions.freestream.velocity         = np.array(np.atleast_1d(speed))
                state.conditions.freestream.mach_number      = np.array(np.atleast_1d(speed / a_delta_ISA))
                state.conditions.freestream.temperature      = np.array(np.atleast_1d(T_delta_ISA))
                state.conditions.freestream.pressure         = np.array(np.atleast_1d(p))
                state.conditions.propulsion.throttle         = np.array(np.atleast_1d(1.))   
            
                results = vehicle.propulsors.turbofan(state) # total thrust
                thrust[idx] = results.thrust_force_vector[0,0]
                mdot[idx]   = results.vehicle_mass_rate[0,0]
            
            scf     = 3600. * mdot / 0.1019715 / thrust
            thrust  = np.divide( thrust , engine_number ) / Units.lbf            
            
            fid.write( '{:7.0f} |'.format(float(altitude/Units.ft)))
            fid.write( np.transpose(map('{:10.1f}'.format,thrust)))
            fid.write( ' |' + 3* ' ') 
            fid.write( np.transpose(map('{:6.3f} '.format,scf)))
            fid.write( '|')
            fid.write( '\n')

                
    # close file
    fid.close
    # Print timestamp
    fid.write('\n\n' + 50*'-' + '\n' + datetime.datetime.now().strftime(" %A, %d. %B %Y %I:%M:%S %p"))
    
    #done! 
    return

# ----------------------------------------------------------------------
#   Module Test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print(' Error: No test defined ! ')    