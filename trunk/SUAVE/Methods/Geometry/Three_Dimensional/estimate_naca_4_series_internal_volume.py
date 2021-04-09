## @ingroup Methods-Geometry-Three_Dimensional
# estimate_naca_4_series_internal_volume.py
#
# Created:  ### ####, M. Vegh
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

## @ingroup Methods-Geometry-Three_Dimensional
def estimate_naca_4_series_internal_volume(wing, m, p): 
    """Computes the volume of a wing with NACA 4-series airfoils.

    Assumptions:
    Wing has constant thickness to chord along the span
    Front spar is at 10%chord and rear spar is at 60% chord

    Source:
    Wikipedia, based on Moran, Jack (2003). An introduction to theoretical and 
      computational aerodynamics. Dover.

    Inputs:
    m                       [-]  percent camber (.1 is 10%)
    p                       [-]  location of max camber (.1 is 10% along chord)
    wing.chords.
      root                  [m]
      tip                   [m]
    wing.taper              [-]
    wing.thickness_to_chord [-]
    wing.spans.projected    [m]
    chord_length            [m]

    Outputs:
    volume                  [m^3] 

    Properties Used:
    N/A
    """          
    #unpack inputs
    t_c   = wing.thickness_to_chord
    taper = wing.taper
    rc    = wing.chords.root
    tc    = wing.chords.tip
    span  = wing.spans.projected

    yc_front_spar_root = m*(.1*rc/p**2.)*(2*p-.1) #front spar is 10%chords
    yc_rear_spar_root  = m*((rc-.6*rc)/(1-p)**2.)*(1+.6-2*p)
    yc_front_spar_tip  = m*(.1*tc/p**2.)*(2*p-.1) 
    yc_rear_spar_tip   = m*((tc-.6*tc)/(1-p)**2.)*(1+.6-2*p)
    
    dy_dx_front_spar = (2.*m/p**2.)*(p-.1) 
    dy_dx_rear_spar  = (2.*m/(1-p)**2.)*(p-.6) 

    theta_front_spar=np.arctan(dy_dx_front_spar)
    theta_rear_spar=np.arctan(dy_dx_rear_spar)
  
    x_front_spar_root  = .1*rc
    x_rear_spar_root   = .6*rc
    yt_front_spar_root = 5*t_c*rc*(.2969*(.1**.5)-.126*.1-.3516*.1**2+.2843*.1**3.-.1015*.1**4.)
    yt_rear_spar_root  = 5*t_c*rc*(.2969*(.6**.5)-.126*.6-.3516*.6**2.+.2843*.6**3.-.1015*.6**4.)
    
    yt_front_spar_tip  = 5*t_c*tc*(.2969*(.1**.5)-.126*.1-.3516*.1**2+.2843*.1**3.-.1015*.1**4.)
    yt_rear_spar_tip   = 5*t_c*tc*(.2969*(.6**.5)-.126*.6-.3516*.6**2.+.2843*.6**3.-.1015*.6**4.)
    
    yu_front_spar_root = yc_front_spar_root+yt_front_spar_root*np.cos(theta_front_spar)
    yl_front_spar_root = yc_front_spar_root-yt_front_spar_root*np.cos(theta_front_spar)
    yu_rear_spar_root  = yc_rear_spar_root+yt_rear_spar_root*np.cos(theta_rear_spar)
    yl_rear_spar_root  = yc_rear_spar_root-yt_rear_spar_root*np.cos(theta_rear_spar)
    
    yu_front_spar_tip  = yc_front_spar_tip+yt_front_spar_tip*np.cos(theta_front_spar)
    yl_front_spar_tip  = yc_front_spar_tip-yt_front_spar_tip*np.cos(theta_front_spar)
    
    yu_rear_spar_tip   = yc_rear_spar_tip+yt_rear_spar_tip*np.cos(theta_rear_spar)
    yl_rear_spar_tip   = yc_rear_spar_tip-yt_rear_spar_tip*np.cos(theta_rear_spar)

    #create line between root and tip points
    slopeu_front_spar  = (yu_front_spar_root-yu_front_spar_tip)/(span/2.)
    slopel_front_spar  = (yl_front_spar_root-yl_front_spar_tip)/(span/2.)
    slopeu_rear_spar   = (yu_rear_spar_root-yu_rear_spar_tip)/(span/2.)
    slopel_rear_spar   = (yl_rear_spar_root-yl_rear_spar_tip)/(span/2.)
    
    yu_front_spar_mid  = (span/4.)*slopeu_front_spar+yu_front_spar_tip
    yl_front_spar_mid  = (span/4.)*slopeu_front_spar+yl_front_spar_tip
    yu_rear_spar_mid   = (span/4.)*slopeu_rear_spar+yu_rear_spar_tip
    yl_rear_spar_mid   = (span/4.)*slopel_rear_spar+yl_rear_spar_tip
    
    slope_chord        = (rc-tc)/(span/2.)
    mid_chord          = slope_chord*(span/4.)+tc
    
    Aroot  = ((yu_front_spar_root-yl_front_spar_root)+(yu_rear_spar_root-yl_rear_spar_root))*(rc*(.6-.1))/2.
    Atip   = ((yu_front_spar_tip-yl_front_spar_tip)+(yu_rear_spar_tip-yl_rear_spar_tip))*(tc*(.6-.1))/2.
    Amid   = ((yu_front_spar_mid-yl_front_spar_mid)+(yu_rear_spar_mid-yl_rear_spar_mid))*(mid_chord*(.6-.1))/2.
    volume = 2*(span/12.)*(Aroot+4*(Amid)+Atip) #integrate a side and multiply by two
    
    return volume