# wave_drag_volume.py
# 
# Created:  Tim MacDonald, 6/24/14
# Modified: Tim MacDonald, 6/24/14
# 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import copy
import numpy as np

# ----------------------------------------------------------------------
#   Wave Drag Volume
# ----------------------------------------------------------------------

def wave_drag_volume(conditions,geometry,flag105):
    """ SUAVE.Methods.wave_drag_volume(conditions,configuration,fuselage)
        computes the wave drag due to lift 
        Based on http://adg.stanford.edu/aa241/drag/ssdragcalc.html
        
        Inputs: total_length, Sref, t/c, Mach

        Outputs:

        Assumptions:

        
    """
    
    import vsp

    # unpack inputs
    freestream   = conditions.freestream
    ref_area     = geometry.reference_area
    tag          = geometry.tag
    
    # conditions
    Mc  = copy.copy(freestream.mach_number)
    cd_w_all = np.zeros(np.shape(Mc))
    vsp.ClearVSPModel()
    vsp.ReadVSPFile(tag+'.vsp3')
    vsp.SetIntAnalysisInput('WaveDrag', 'NumSlices', [30])
    vsp.SetIntAnalysisInput('WaveDrag', 'NumRotSects', [15]) 
    
    if flag105 is True:
        vsp.SetDoubleAnalysisInput('WaveDrag', 'Mach', [1.05])    
        ridwd = vsp.ExecAnalysis('WaveDrag')        
        cd_w = vsp.GetDoubleResults(ridwd,'CDWave')
        cd_w = cd_w[0]*100./ref_area # default ref area in VSP doesn't seem to have an easy change
        return cd_w
    
    for ii,mach in enumerate(Mc):
        if mach >= 1.05:
            vsp.SetDoubleAnalysisInput('WaveDrag', 'Mach', [float(mach)])
            ridwd = vsp.ExecAnalysis('WaveDrag') 
            cd_w = vsp.GetDoubleResults(ridwd,'CDWave')
            cd_w = cd_w[0]*100./ref_area
            cd_w_all[ii] = cd_w
    
    return cd_w_all