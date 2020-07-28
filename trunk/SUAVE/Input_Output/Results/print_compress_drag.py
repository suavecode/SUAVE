## @ingroup Input_Output-Results
#print_compress_drag.py

# Created: SUAVE team
# Modified: Carlos Ilario, Feb 2016
#           Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Core import Units,Data

# Imports
import time                     # importing library
import datetime                 # importing library

# ----------------------------------------------------------------------
#  Print output file with compressibility drag
# ----------------------------------------------------------------------
## @ingroup Input_Output-Results
def print_compress_drag(vehicle,analyses,filename = 'compress_drag.dat'):
    """This creates a file showing a breakdown of compressibility drag for the vehicle.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    vehicle.wings.main_wing.
      sweeps.quarter_chord     [-]
    vehicle.wings.*.
      tag                     <string>
      thickness_to_chord      [-]
    vehicle.
      tag                     <string>
      reference_area          [m^2]
    analyses.configs.cruise.aerodynamics.settings    Used in called function:
    analyses.configs.cruise.aerodynamics.process.compute.drag.compressibility.wings.wing(state,settings,wing)
    filename                  Sets file name to save (optional)
    

    Outputs:
    filename                  Saved file with name as above

    Properties Used:
    N/A
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
    fid.write(str(np.insert(np.transpose(list(map('M={:5.3f} | '.format,(mach_vec)))),0,'  CL   |  ')))
    fid.write('\n')

    # call aerodynamic method for each CL
    for idcl, cl in enumerate(cl_vec):
        state.conditions.aerodynamics.lift_breakdown.compressible_wings   = Data()
        for wing in vehicle.wings:
            state.conditions.aerodynamics.lift_breakdown.compressible_wings[wing.tag] = np.atleast_1d(cl) 
            analyses.configs.cruise.aerodynamics.process.compute.drag.compressibility.wings.wing(state,settings,wing)
        # process output for print
        drag_breakdown = state.conditions.aerodynamics.drag_breakdown.compressible
        for wing in vehicle.wings:
            cd_compress[wing.tag][:,idcl] =  drag_breakdown[wing.tag].compressibility_drag
            cd_compress_tot[:,idcl]      +=  drag_breakdown[wing.tag].compressibility_drag
        # print first the TOTAL COMPRESSIBILITY DRAG    
        fid.write(str(np.insert((np.transpose(list(map('{:7.5f} | '.format,(cd_compress_tot[:,idcl]))))),0,' {:5.3f} |  '.format(cl))))
        fid.write('\n')
    fid.write( 119*'-' )
    # print results of other components
    for wing in vehicle.wings: 
        fid.write('\n ' + wing.tag.upper() + '  ( t/c: {:4.3f} )'.format(wing.thickness_to_chord) + '\n')
        fid.write(str(np.insert(np.transpose(list(map('M={:5.3f} | '.format,(mach_vec)))),0,'  CL   |  ')))
        fid.write('\n')
        for idcl, cl in enumerate(cl_vec):
            fid.write(str(np.insert((np.transpose(list(map('{:7.5f} | '.format,(cd_compress[wing.tag][:,idcl]))))),0,' {:5.3f} |  '.format(cl))))
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