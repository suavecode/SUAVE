## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
# induced_drag_aircraft.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Jan 2016, E. Botero
#           Apr 2020, M. Clarke     
#           Jun 2020, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Core import Data

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Induced Drag Aircraft
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
def induced_drag_aircraft(state,settings,geometry):
    """Determines induced drag for the full aircraft

    Assumptions:
    Based on fits

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)

    Inputs:
    state.conditions.aerodynamics.lift_coefficient               [Unitless]
    state.conditions.aerodynamics.drag_breakdown.parasite.total  [Unitless]
    configuration.oswald_efficiency_factor                       [Unitless]
    configuration.viscous_lift_dependent_drag_factor             [Unitless]
    geometry.wings['main_wing'].span_efficiency                  [Unitless]
    geometry.wings['main_wing'].aspect_ratio                     [Unitless]

    Outputs:
    total_induced_drag                                           [Unitless]

    Properties Used:
    N/A
    """

    # unpack inputs
    conditions    = state.conditions
    configuration = settings
    wings         = geometry.wings
    
    K       = configuration.viscous_lift_dependent_drag_factor
    e_osw   = configuration.oswald_efficiency_factor	
    CL      = conditions.aerodynamics.lift_coefficient
    CDi     = conditions.aerodynamics.drag_breakdown.induced.inviscid
    S_ref   = geometry.reference_area

    wing_viscous_induced_drags = Data()

    # If the oswald efficiency factor is not specified (typical case)
    if e_osw == None:

        # Prime totals
        area                        = 0.
        AR                          = 0.01  
        total_induced_viscous_drag  = 0.
        
        # Go through each wing, and make calculations
        for wing in wings:
            tag        = wing.tag
            ar         = wing.aspect_ratio
            s_wing     = conditions.aerodynamics.drag_breakdown.parasite[wing.tag].reference_area
            cdp_wing   = conditions.aerodynamics.drag_breakdown.parasite[tag].parasite_drag_coefficient
            cl_wing    = conditions.aerodynamics.lift_breakdown.inviscid_wings[tag]
                
            cdi_v_wing = K*cdp_wing*(cl_wing**2)
            total_induced_viscous_drag  = total_induced_viscous_drag  + (cdi_v_wing)*(s_wing/S_ref)
            
            # pack the wing level viscous induced drag
            wing_viscous_induced_drags[tag] = cdi_v_wing 
            
            # Save this for later
            if s_wing > area:
                area = s_wing
                AR  = ar
                
        total_induced_drag = total_induced_viscous_drag + CDi
        
        # Now calculate the vehicle level oswald efficiency
        e_osw = (CL**2)/(np.pi*AR*total_induced_drag)
        
        
    # If the user specifies a vehicle level oswald efficiency factor
    else:
        
        # Find the largest wing, use that for AR
        S  = 0.
        AR = 0.01        
        for wing in wings:
            if wing.areas.reference>S:
                AR = wing.aspect_ratio
                
        # Calculate the induced drag       
        total_induced_drag = CL **2 / (np.pi*AR*e_osw)
        total_induced_viscous_drag = total_induced_drag - CDi
    
    
    conditions.aerodynamics.drag_breakdown.induced.total              = total_induced_drag
    conditions.aerodynamics.drag_breakdown.induced.viscous            = total_induced_viscous_drag
    conditions.aerodynamics.drag_breakdown.induced.efficiency_factor  = e_osw
    conditions.aerodynamics.drag_breakdown.induced.viscous_wings_drag = wing_viscous_induced_drags
    
    return total_induced_drag