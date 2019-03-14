## @ingroup Methods-Weights-Correlations-Common
# wing_segmented_planform.py
# 
# Created:  Mar 2019, E. Botero 
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Data

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def wing_segmented_planform(wing):
    """Computes standard wing planform values.
    
    Assumptions:
    Multisigmented wing. We only find the first spanwise location of the mean aerodynamic chord
    
    Source:
    None
    
    Inputs:
    wing.
      chords.root              [m]
      spans.projected          [m]
      vertical                 <boolean> Determines if wing is vertical
      symmetric                <boolean> Determines if wing is symmetric
    
    Outputs:
    wing.
      spans.total              [m]
      chords.tip               [m]
      chords.mean_aerodynamics [m]
      areas.reference          [m^2]
      taper                    [-]
      sweeps.quarter_chord     [radians]
      aspect_ratio             [-]
      thickness_to_chord       [-]
      dihedral                 [radians]
      
      aerodynamic_center       [m]      x, y, and z location

        
    
    Properties Used:
    N/A
    """
    
    # Unpack
    span = wing.spans.projected
    RC   = wing.chords.root
    
    # Pull all the segment data into array format
    span_locs = []
    twists    = []
    sweeps    = []
    dihedrals = []
    chords    = []
    t_cs      = []
    for key in wing.Segments:
        seg = wing.Segments[key]
        span_locs.append(seg.percent_span_location)
        twists.append(seg.twist)
        chords.append(seg.root_chord_percent)
        sweeps.append(seg.sweeps.quarter_chord)
        t_cs.append(seg.thickness_to_chord)
        dihedrals.append(seg.dihedral_outboard)
        
    # Convert to arrays
    chords    = np.array(chords)
    span_locs = np.array(span_locs)
    sweeps    = np.array(sweeps)
    
    # Basic calcs:
    lengths = span_locs[1:]-span_locs[:-1]
    
    # Calculate the areas of each segment
    As = (span/2)*RC*((lengths)*chords[:-1]-(chords[:-1]-chords[1:])*(lengths)/2)
    
    # Calculate the wing area
    ref_area = np.sum(As)*2
    
    # Calculate the Aspect Ratio
    AR = (span**2)/ref_area
    
    # Calculate the total span
    lens = span*(lengths)/np.cos(dihedrals[:-1])
    total_len = np.sum(np.array(lens))
    
    # Calculate the mean geometric chord
    mgc = ref_area/span
    
    # Calculate the mean aerodynamic chord
    A = RC*chords[:-1]
    B = (A-RC*chords[1:])/(span_locs[:-1]-span_locs[1:])
    C = span_locs[:-1]
    integral = ((A+B*(span_locs[1:]-C))**3-(A+B*(span_locs[:-1]-C))**3)/(3*B)
    # For the cases when the wing doesn't taper in a spot
    integral[np.isnan(integral)] = (A[np.isnan(integral)]**2)*((lengths)[np.isnan(integral)])
    MAC = (span/(ref_area))*np.sum(integral)
    
    # Compute MAC Location, this will do the first location
    mac_percent     = MAC/RC
    if len(np.where(chords>mac_percent)[0])>0:
        i               = np.where(chords>mac_percent)[0][-1]
        mac_loc_non_dim = (chords[i]-mac_percent)*(span_locs[i+1]-span_locs[i])/(chords[i]-chords[i+1]) + span_locs[i]
    else: # This is a non tapered wing
        mac_loc_non_dim = 0.5
    
    # Calculate the effective taper ratio
    lamda = 2*mgc/RC - 1
    
    # effective tip chord
    ct = lamda*RC
    
    # Calculate an average t/c weighted by area
    t_c = np.sum(As*t_cs[:-1])/(ref_area/2)
    
    # Calculate the segment leading edge sweeps
    r_offsets = RC*chords[:-1]/4
    t_offsets = RC*chords[1:]/4
    le_sweeps = np.arctan((r_offsets+np.tan(sweeps[:-1])*(lengths*span/2)-t_offsets)/(lengths*span/2))    
    
    # Calculate the effective sweeps
    c_4_sweep   = np.arctan(np.sum(lengths*np.tan(sweeps[:-1])))
    le_sweep_total= np.arctan(np.sum(lengths*np.tan(le_sweeps)))

    # Calculate the aerodynamic center, but first the centroid
    dxs = np.concatenate([np.array([0]),np.tan(le_sweeps[:-1])*lengths[:-1]*span/2])
    dys = np.concatenate([np.array([0]),lengths[:-1]*span/2])
    
    Cxys = []
    for i in range(len(lengths)):
        Cxys.append(segment_centroid(le_sweeps[i], span*lengths[i]/2, dxs[i], dys[i], chords[i]*RC, chords[i+1]*RC, As[i]))

    total_centroid = np.sum(Cxys*As,axis=0)/(ref_area/2)
    
    aerodynamic_center = [total_centroid[0]-MAC/4,total_centroid[1],0]
    
    # Total length for supersonics
    total_length = np.tan(le_sweep_total)*span/2 + chords[-1]*RC

    
    # Pack stuff
    wing.areas.reference         = ref_area
    wing.aspect_ratio            = AR
    wing.spans.total             = total_len
    wing.chords.mean_geometric   = mgc
    wing.chords.mean_aerodynamic = MAC
    wing.chords.tip              = ct
    wing.taper                   = lamda
    wing.sweeps.quarter_chord    = c_4_sweep
    wing.sweeps.leading_edge     = le_sweep_total
    wing.thickness_to_chord      = t_c
    wing.aerodynamic_center      = aerodynamic_center
    wing.total_length            = total_length
    
    return wing

# Segment centroid
def segment_centroid(seg_le_sweep,seg_semispan,dx,dy,seg_rc,seg_tc,A):
    """Computes the centroid of a polygonal segment
    
    Assumptions:
    Polygon
    
    Source:
    None
    
    Inputs:
    seg_le_sweep  [rad]
    seg_semispan  [m]
    dx            [m]
    dy            [m]
    seg_rc        [m]
    seg_tc        [m]
    A             [m**2]

    Outputs:
    cx,cy        [m,m]

    Properties Used:
    N/A
    """    
    xi = np.array([0,np.tan(seg_le_sweep)*seg_semispan,np.tan(seg_le_sweep)*seg_semispan+seg_tc,seg_rc])
    yi = np.array([0,seg_semispan,seg_semispan,0])
    
    cx = -np.sum((xi[:-1]+xi[1:])*(xi[:-1]*yi[1:]-xi[1:]*yi[:-1]))/(6*A)+dx
    cy = -np.sum((yi[:-1]+yi[1:])*(xi[:-1]*yi[1:]-xi[1:]*yi[:-1]))/(6*A)+dy
    
    return np.array([cx,cy])