## @ingroup Input_Output-OpenVSP
# write_vsp_fea.py
# 
# Created:  Mar 2018, T. MacDonald
# Modified: 

try:
    import vsp_g as vsp
except ImportError:
    pass # This allows SUAVE to build without OpenVSP
import numpy as np
import time
import fileinput

## @ingroup Input_Output-OpenVSP
def run_vspaero(vehicle,tag,vspaero_settings):
    """
    
    Assumptions:
    None

    Source:
    N/A

    Inputs:

    Outputs:                             

    Properties Used:
    N/A
    """
    
    alpha_start = vspaero_settings.alpha_start
    alpha_end   = vspaero_settings.alpha_end
    alpha_npts  = vspaero_settings.alpha_npts
    
    # VSPAERO Result Output Indicies, may change over time
    AoA_ind = 2
    CL_ind  = 4
    
    #alpha_start = 0
    #alpha_end   = 20 # deg
    #alpha_npts  = 1
    
    # Reset OpenVSP to avoid including a previous vehicle
    vsp.ClearVSPModel()    
    
    vsp.ReadVSPFile(tag + '.vsp3')
    vehicle_id = vsp.FindContainersWithName('Vehicle')[0]
    
    analysis_name = 'VSPAEROSweep'
    Sref = vehicle.reference_area
    
    vsp.SetDoubleAnalysisInput(analysis_name,'Sref',[Sref])
    vsp.SetDoubleAnalysisInput(analysis_name,'AlphaStart',[alpha_start])
    vsp.SetDoubleAnalysisInput(analysis_name,'AlphaEnd',[alpha_end])
    vsp.SetIntAnalysisInput(analysis_name,'AlphaNpts',[alpha_npts])
    vsp.SetDoubleAnalysisInput(analysis_name,'MachStart',[0.2])
    
    results_id = vsp.ExecAnalysis(analysis_name)
    
    vsp.WriteResultsCSVFile(results_id,'')

    output_file = tag+'_DegenGeom.polar'
    fi = open(output_file)
    
    num_runs = alpha_npts
    num_outputs = 2 # to be changed based on modifications to desired outputs
    CLs  = np.zeros(num_runs)
    AoAs = np.zeros(num_runs)
    for i,line in enumerate(fi):
        if i == 0:
            pass
        else:
            run_ind = i-1
            AoAs[run_ind] = line.split()[AoA_ind]
            CLs[run_ind]  = line.split()[CL_ind]
            
    return AoAs,CLs
    
if __name__ == '__main__':
    tag = '/home/tim/Documents/Experimental/OpenVSP_VSPAERO_Test/aero_test'
    import sys
    sys.path.append('/home/tim/Documents/Experimental/OpenVSP_VSPAERO_Test')
    from Concorde import vehicle_setup, configs_setup    
    vehicle = vehicle_setup()
    AoAs, CLs = run_vspaero(vehicle,tag)
    # Concorde real values from AIAA case study
    AoA_real = np.array([0,5,10,15,20])
    CL_real = np.array([0.,.17,.43,.73,1.03])    
    
    import pylab as plt
    from SUAVE.Core import Units
    
    fig = plt.figure("Drag Components",figsize=(8,6))
    axes = plt.gca()
    axes.plot( AoAs, CLs , 'bo--')  
    axes.plot( AoA_real, CL_real , 'ro--')  

    axes.set_xlabel('Angle of Attack (deg)')
    axes.set_ylabel('$C_L$')
    axes.set_title('$C_L$ v AoA at 0 ft, Mach 0.2')
    axes.legend(['VSPAERO $C_L$','Concorde Real $C_L$'])
    plt.minorticks_on()
    plt.axes().grid(which='both')
    plt.axes().grid(which='minor',lw=.5)
    plt.axes().grid(which='major',lw=1.5)  
    
    plt.show()
    
    a = 0