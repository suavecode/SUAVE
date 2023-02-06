## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Planform
# wing_fuel_volume.py
#
# Created:  Apr 2014, T. Orra
# Modified: Sep 2016, E. Botero

# ----------------------------------------------------------------------
#  Correlation-based methods for wing fuel capacity estimation
# ----------------------------------------------------------------------
## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Planform
def wing_fuel_volume(wing):
    """Calculates the available fuel volume in a wing.

    Assumptions:
    None

    Source:
    Torenbeek, E., "Advanced Aircraft Design", 2013 (equation 10.30)

    Inputs:
    wing.
      areas.reference    [m^2]
      aspect_ratio       [-]
      thickness_to_chord [-]

    Outputs:
    wing.volume          [m^3]

    Properties Used:
    N/A
    """              

    # Unpack
    sref  = wing.areas.reference
    ar    = wing.aspect_ratio
    tc    = wing.thickness_to_chord

    # Calculate
    volume = 0.90* tc * sref** 1.5 * ar**-0.5 * 0.55

    # Pack
    wing.fuel_volume = volume