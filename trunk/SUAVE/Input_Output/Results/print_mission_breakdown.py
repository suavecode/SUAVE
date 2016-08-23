# print_mission_breakdown.py

# Created: SUAVE team
# Updated: Carlos Ilario, Feb 2016

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np

from SUAVE.Core import Units
import time                     # importing library
import datetime                 # importing library

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def print_mission_breakdown(results,filename='mission_breakdown.dat'):
    """ SUAVE.Methods.Results.mission_breakdown(results,filename='mission_breakdown.dat'):
        
        Print output file with compressibility drag
        
        Inputs:
            results - Data dictionary with the fields:
                ? ?? ? ? ?
            filename [optional] - Name of the file to be created

        Outputs:
            output file

        Assumptions:

    """ 

    fid = open(filename,'w')   # Open output file
    fid.write('Output file with mission profile breakdown\n\n') #Start output printing

    k1 = 1.727133242E-06                            # constant to airspeed conversion
    k2 = 0.2857142857                               # constant to airspeed conversion

    TotalRange = 0
    i = 0
    for key in results.segments.keys():        #loop for all segments
        segment = results.segments[key]

        HPf = -segment.conditions.frames.inertial.position_vector[-1,2] / Units.ft      #Final segment Altitude   [ft]
        HPi = -segment.conditions.frames.inertial.position_vector[0,2] / Units.ft       #Initial segment Altitude  [ft]

        CLf = segment.conditions.aerodynamics.lift_coefficient[-1]     #Final Segment CL [-]
        CLi = segment.conditions.aerodynamics.lift_coefficient[0]      #Initial Segment CL [-]
        Tf  =  segment.conditions.frames.inertial.time[-1]/ Units.min   #Final Segment Time [min]
        Ti  =  segment.conditions.frames.inertial.time[0] / Units.min   #Initial Segment Time [min]
        Wf  =  segment.conditions.weights.total_mass[-1]                  #Final Segment weight [kg]
        Wi  =  segment.conditions.weights.total_mass[0]                   #Initial Segment weight [kg]
        Dist = (segment.conditions.frames.inertial.position_vector[-1,0] - segment.conditions.frames.inertial.position_vector[0,0] ) / Units.nautical_miles #Distance [nm]
        TotalRange = TotalRange + Dist

        Mf = segment.conditions.freestream.mach_number[-1]          # Final segment mach number
        Mi = segment.conditions.freestream.mach_number[0]           # Initial segment mach number

#       Aispeed conversion: KTAS to  KCAS
        atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        p0 , dummy , dummy , dummy , dummy  = atmosphere.compute_values(0)
        deltai = segment.conditions.freestream.pressure[0] / p0
        deltaf = segment.conditions.freestream.pressure[-1]/ p0

        VEi = Mi*(340.294*np.sqrt(deltai))          #Equivalent airspeed [m/s]
        QCPOi = deltai*((1.+ k1*VEi**2/deltai)**3.5-1.) #
        VCi = np.sqrt(((QCPOi+1.)**k2-1.)/k1)       #Calibrated airspeed [m/s]
        KCASi = VCi / Units.knots                   #Calibrated airspeed [knots]

        VEf = Mf*(340.294*np.sqrt(deltaf))          #Equivalent airspeed [m/s]
        QCPOf = deltaf*((1.+ k1*VEf**2/deltaf)**3.5-1.)
        VCf = np.sqrt(((QCPOf+1.)**k2-1.)/k1) #m/s  #Calibrated airspeed [m/s]
        KCASf = VCf / Units.knots                   #Calibrated airspeed [knots]

#       String formatting
        CLf_str =   str('%15.3f'   % CLf)     + '|'
        CLi_str =   str('%15.3f'   % CLi)     + '|'
        HPf_str =   str('%7.0f'    % HPf)     + '|'
        HPi_str =   str('%7.0f'    % HPi)     + '|'
        Dist_str =  str('%9.0f'    % Dist)    + '|'
        Wf_str =    str('%8.0f'    % Wf)      + '|'
        Wi_str =    str('%8.0f'    % Wi)      + '|'
        T_str =     str('%7.1f'   % (Tf-Ti))  + '|'
        Fuel_str=   str('%8.0f'   % (Wi-Wf))  + '|'
        Mi_str =    str('%7.3f'   % Mi)       + '|'
        Mf_str =    str('%7.3f'   % Mf)       + '|'
        KCASi_str = str('%7.1f'   % KCASi)    + '|'
        KCASf_str = str('%7.1f'   % KCASf)    + '|'

        Segment_str = '%- 31s |' % key 
        

        if i == 0:  #Write header
            fid.write( '         FLIGHT PHASE           |   ALTITUDE    |     WEIGHT      |  DIST.  | TIME  |  FUEL  |            SPEED              |\n')
            fid.write( '                                | From  |   To  |Inicial | Final  |         |       |        |Inicial| Final |Inicial| Final |\n')
            fid.write( '                                |   ft  |   ft  |   kg   |   kg   |    nm   |  min  |   kg   | KCAS  | KCAS  |  Mach |  Mach |\n')
            fid.write( '                                |       |       |        |        |         |       |        |       |       |       |       |\n')
        # Print segment data
        fid.write( Segment_str+HPi_str+HPf_str+Wi_str+Wf_str+Dist_str+T_str+Fuel_str+KCASi_str+KCASf_str+Mi_str+Mf_str+'\n')
        i = i+1

    #Summary of results [nm]
    TotalFuel = results.segments[0].conditions.weights.total_mass[0] - results.segments[-1].conditions.weights.total_mass[-1]   #[kg]
    TotalTime = (results.segments[-1].conditions.frames.inertial.time[-1] - results.segments[0].conditions.frames.inertial.time[0])  #[min]

    fid.write(2*'\n')
    fid.write(' Total Range (nm) ........... '+ str('%9.0f'   % TotalRange)+'\n')
    fid.write(' Total Fuel  (kg) ........... '+ str('%9.0f'   % TotalFuel)+'\n')
    fid.write(' Total Time  (hh:mm) ........ '+ time.strftime('    %H:%M', time.gmtime(TotalTime))+'\n')
    # Print timestamp
    fid.write(2*'\n'+ 43*'-'+ '\n' + datetime.datetime.now().strftime(" %A, %d. %B %Y %I:%M:%S %p"))
    
    fid.close

    #done! 
    return

# ----------------------------------------------------------------------
#   Module Test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print(' Error: No test defined ! ')    