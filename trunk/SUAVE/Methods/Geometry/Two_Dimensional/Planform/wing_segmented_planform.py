## @ingroup Methods-Weights-Correlations-Common
# wing_segmented_planform.py
# 
# Created:  Mar 2019, E. Botero 
# Modified: Feb 2021, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Data

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def wing_segmented_planform(wing, overwrite_reference = False):
    """Computes standard wing planform values.
    
    Assumptions:
    Multisegmented wing. There is no unexposed wetted area, ie wing area that 
    intersects inside a fuselage. Aerodynamic center is at 25% mean aerodynamic chord.
    
    Source:
    None
    
    Inputs:
    overwrite_reference        <boolean> Determines if reference area, wetted area, and aspect
                                         ratio are overwritten based on the segment values.
    wing.
      chords.root              [m]
      spans.projected          [m]
      symmetric                <boolean> Determines if wing is symmetric
    
    Outputs:
    wing.
      spans.total                [m]
      chords.tip                 [m]
      chords.mean_aerodynamics   [m]
      wing.chords.mean_geometric [m]
      areas.reference            [m^2]
      taper                      [-]
      sweeps.quarter_chord       [radians]
      aspect_ratio               [-]
      thickness_to_chord         [-]
      dihedral                   [radians]
      
      aerodynamic_center         [m]      x, y, and z location

        
    
    Properties Used:
    N/A
    """
    
    # Unpack
    span = wing.spans.projected
    RC   = wing.chords.root
    sym  = wing.symmetric
    
    # Pull all the segment data into array format
    span_locs = []
    twists    = []
    sweeps    = []
    dihedrals = []
    chords    = []
    t_cs      = []
    for key in wing.Segments.keys():
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
    t_cs      = np.array(t_cs)
    
    # Basic calcs:
    semispan     = span/(1+sym)
    lengths_ndim = span_locs[1:]-span_locs[:-1]
    lengths_dim  = lengths_ndim*semispan
    chords_dim   = RC*chords
    tapers       = chords[1:]/chords[:-1]
    
    # Calculate the areas of each segment
    As = (lengths_dim*chords_dim[:-1]-(chords_dim[:-1]-chords_dim[1:])*(lengths_dim/2))
    
    # Calculate the weighted area, this should not include any unexposed area 
    A_wets = 2*(1+0.2*t_cs[:-1])*As
    wet_area = np.sum(A_wets)
    
    # Calculate the wing area
    ref_area = np.sum(As)*(1+sym)
    
    # Calculate the Aspect Ratio
    AR = (span**2)/ref_area
    
    # Calculate the total span
    lens = lengths_dim/np.cos(dihedrals[:-1])
    total_len = np.sum(np.array(lens))*(1+sym)
    
    # Calculate the mean geometric chord
    mgc = ref_area/span
    
    # Calculate the mean aerodynamic chord
    A = chords_dim[:-1]
    B = (A-chords_dim[1:])/(-lengths_ndim)
    C = span_locs[:-1]
    integral = ((A+B*(span_locs[1:]-C))**3-(A+B*(span_locs[:-1]-C))**3)/(3*B)
    # For the cases when the wing doesn't taper in a spot
    integral[np.isnan(integral)] = (A[np.isnan(integral)]**2)*((lengths_ndim)[np.isnan(integral)])
    MAC = (semispan*(1+sym)/(ref_area))*np.sum(integral)
    
    # Calculate the taper ratio
    lamda = chords[-1]/chords[0]
    
    # the tip chord
    ct = chords_dim[-1]
    
    # Calculate an average t/c weighted by area
    t_c = np.sum(As*t_cs[:-1])/(ref_area/2)
    
    # Calculate the segment leading edge sweeps
    r_offsets = chords_dim[:-1]/4
    t_offsets = chords_dim[1:]/4
    le_sweeps = np.arctan((r_offsets+np.tan(sweeps[:-1])*(lengths_dim)-t_offsets)/(lengths_dim))    
    
    # Calculate the effective sweeps
    c_4_sweep   = np.arctan(np.sum(lengths_ndim*np.tan(sweeps[:-1])))
    le_sweep_total= np.arctan(np.sum(lengths_ndim*np.tan(le_sweeps)))

    # Calculate the aerodynamic center, but first the centroid
    dxs = np.cumsum(np.concatenate([np.array([0]),np.tan(le_sweeps[:-1])*lengths_dim[:-1]]))
    dys = np.cumsum(np.concatenate([np.array([0]),lengths_dim[:-1]]))
    dzs = np.cumsum(np.concatenate([np.array([0]),np.tan(dihedrals[:-2])*lengths_dim[:-1]]))
    
    Cxys = []
    for i in range(len(lengths_dim)):
        Cxys.append(segment_centroid(le_sweeps[i],lengths_dim[i],dxs[i],dys[i],dzs[i], tapers[i], 
                                     As[i], dihedrals[i], chords_dim[i], chords_dim[i+1]))

    aerodynamic_center = (np.dot(np.transpose(Cxys),As)/(ref_area/(1+sym)))

    single_side_aerodynamic_center = (np.array(aerodynamic_center)*1.)
    single_side_aerodynamic_center[0] = single_side_aerodynamic_center[0] - MAC*.25    
    if sym== True:
        aerodynamic_center[1] = 0
        
    aerodynamic_center[0] = single_side_aerodynamic_center[0]
    
    # Total length for supersonics
    total_length = np.tan(le_sweep_total)*semispan + chords[-1]*RC
    
    # Pack stuff
    if overwrite_reference:
        wing.areas.reference         = ref_area
        wing.areas.wetted            = wet_area
        wing.aspect_ratio            = AR

    wing.spans.total                    = total_len
    wing.chords.mean_geometric          = mgc
    wing.chords.mean_aerodynamic        = MAC
    wing.chords.tip                     = ct
    wing.taper                          = lamda
    wing.sweeps.quarter_chord           = c_4_sweep
    wing.sweeps.leading_edge            = le_sweep_total
    wing.thickness_to_chord             = t_c
    wing.aerodynamic_center             = aerodynamic_center
    wing.single_side_aerodynamic_center = single_side_aerodynamic_center
    wing.total_length                   = total_length
    
    return wing

# Segment centroid
def segment_centroid(le_sweep,seg_span,dx,dy,dz,taper,A,dihedral,root_chord,tip_chord):
    """Computes the centroid of a trapezoidal segment
    
    Assumptions:
    Polygon
    
    Source:
    None
    
    Inputs:
    le_sweep      [rad]
    seg_span      [m]
    dx            [m]
    dy            [m]
    taper         [dimensionless]
    A             [m**2]
    dihedral      [radians]
    root_chord    [m]
    tip_chord     [m]

    Outputs:
    cx,cy         [m,m]

    Properties Used:
    N/A
    """    
    
    a = tip_chord
    b = root_chord
    c = np.tan(le_sweep)*seg_span
    cx = (2*a*c + a**2 + c*b + a*b + b**2) / (3*(a+b))
    cy = seg_span / 3. * (( 1. + 2. * taper ) / (1. + taper))
    cz = cy * np.tan(dihedral)    
    
    return np.array([cx+dx,cy+dy,cz+dz])