#print_compress_drag.py

# Created: SUAVE team
# Updated: Carlos Ilario, Feb 2016

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Core import Units,Data

# Imports
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Drag.compressibility_drag_wing import compressibility_drag_wing
import time                     # importing library
import datetime                 # importing library

# ----------------------------------------------------------------------
#  Print output file with compressibility drag
# ----------------------------------------------------------------------
def print_compress_drag(vehicle,analyses,filename = 'compress_drag.dat'):
    """ SUAVE.Methods.Results.print_compress_drag(vehicle,filename = 'compress_drag.dat'):
        
        Print output file with compressibility drag
        
        Inputs:
            vehicle         - SUave type vehicle
            analyses        - 
            filename [optional] - Name of the file to be created

        Outputs:
            output file

        Assumptions:

    """ 

    # Unpack
    sweep           = vehicle.wings['main_wing'].sweeps.quarter_chord  / Units.deg
    t_c             = vehicle.wings['main_wing'].thickness_to_chord
    sref            = vehicle.reference_area
    settings        = analyses.configs.cruise.aerodynamics.settings
    
    # Define mach and CL vectors    
    mach_vec                = np.linspace(0.45,0.95,11)
    cl_vec                  = np.linspace(0.30,0.80,11)    
    # allocating array for each wing
    cd_compress = Data()
    for idw,wing in enumerate(vehicle.wings):
        cd_compress[wing.tag] = np.zeros((len(mach_vec),len(cl_vec)))
    cd_compress_tot = np.zeros_like(cd_compress.main_wing)        
    
    # Alocatting array necessary for the drag estimation method
    state = Data()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    state.conditions.freestream.mach_number      = mach_vec

    # write header of file
    fid = open(filename,'w')   # Open output file
    fid.write('Output file with compressibility drag breakdown\n\n') 
    fid.write('  VEHICLE TAG : ' + vehicle.tag + '\n\n')
    fid.write('  REFERENCE AREA ................ ' + str('%5.1f' %   sref               )   + ' m2 ' + '\n')
    fid.write('  MAIN WING SWEEP ............... ' + str('%5.1f' %   sweep              )   + ' deg' + '\n')
    fid.write('  MAIN WING THICKNESS RATIO ..... ' + str('%5.2f' %   t_c                )   + '    ' + '\n')
    fid.write(' \n')
    fid.write(' TOTAL COMPRESSIBILITY DRAG \n')
    fid.write(np.insert(np.transpose(map('M={:5.3f} | '.format,(mach_vec))),0,'  CL   |  '))
    fid.write('\n')

    # call aerodynamic method for each CL
    for idcl, cl in enumerate(cl_vec):
        state.conditions.aerodynamics.lift_breakdown.compressible_wings = np.atleast_1d(cl)
        # call method
        for wing in vehicle.wings:
            analyses.configs.cruise.aerodynamics.process.compute.drag.compressibility.wings.wing(state,settings,wing)
        # process output for print
        drag_breakdown = state.conditions.aerodynamics.drag_breakdown.compressible
        for wing in vehicle.wings:
            cd_compress[wing.tag][:,idcl] =  drag_breakdown[wing.tag].compressibility_drag
            cd_compress_tot[:,idcl]      +=  drag_breakdown[wing.tag].compressibility_drag
        # print first the TOTAL COMPRESSIBILITY DRAG    
        fid.write(np.insert((np.transpose(map('{:7.5f} | '.format,(cd_compress_tot[:,idcl])))),0,' {:5.3f} |  '.format(cl)))
        fid.write('\n')
    fid.write( 119*'-' )
    # print results of other components
    for wing in vehicle.wings: 
        fid.write('\n ' + wing.tag.upper() + '  ( t/c: {:4.3f} )'.format(wing.thickness_to_chord) + '\n')
        fid.write(np.insert(np.transpose(map('M={:5.3f} | '.format,(mach_vec))),0,'  CL   |  '))
        fid.write('\n')
        for idcl, cl in enumerate(cl_vec):
            fid.write(np.insert((np.transpose(map('{:7.5f} | '.format,(cd_compress[wing.tag][:,idcl])))),0,' {:5.3f} |  '.format(cl)))
            fid.write('\n')
        fid.write(119*'-')
    # close file
    fid.close
    # Print timestamp
    fid.write('\n\n' + datetime.datetime.now().strftime(" %A, %d. %B %Y %I:%M:%S %p"))
    
    #done! 
    return

# ----------------------------------------------------------------------
#   Module Test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print(' Error: No test defined ! ')    