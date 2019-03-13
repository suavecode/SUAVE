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
    Multisigmented wing
    
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
      areas.wetted             [m^2]
      areas.affected           [m^2]
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
        t_cs.append(seg.thickness_to_chord)
        dihedrals.append(seg.dihedral_outboard)
        
    # Convert to arrays
    chords    = np.array(chords)
    span_locs = np.array(span_locs)
    
    # Calculate the areas of each segment
    As = span*RC*((span_locs[1:]-span_locs[:-1])*chords[:-1]-(chords[:-1]-chords[1:])*(span_locs[1:]-span_locs[:-1])/2)
    
    # Calculate the wing area
    ref_area = np.sum(As)
    
    # Calculate the Aspect Ratio
    AR = (span**2)/ref_area
    
    # Calculate the total span
    lens = span*(span_locs[1:]-span_locs[:-1])/np.cos(dihedrals[:-1])
    total_len = np.sum(np.array(lens))
    
    # Calculate the mean geometric chord
    mgc = ref_area/span
    
    # Calculate the mean aerodynamic chord
    A = RC*chords[:-1]
    B = (A-RC*chords[1:])/(span_locs[:-1]-span_locs[1:])
    C = span_locs[:-1]
    integral = ((A+B*(span_locs[1:]-C))**3-(A+B*(span_locs[:-1]-C))**3)/(3*B)
    integral[np.isnan(integral)] = (A[np.isnan(integral)]**3)/3
    MAC = (span/(ref_area))*np.sum(integral)
    
    # Calculate the effective taper ratio
    lamda = 2*mgc/RC -1
    
    # effective tip chord
    ct = lamda*RC
    
    # Calculate the aerodynamic_center
    
    # Pack stuff
    wing.areas.reference         = ref_area
    wing.aspect_ratio            = AR
    wing.spans.total             = total_len
    wing.chords.mean_geometric   = mgc
    wing.chords.mean_aerodynamic = MAC
    wing.chords.tip              = ct
    
    return wing