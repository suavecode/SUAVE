# APC_10x7_thin_electric.py
#
# Created:  Dec 2020, R. Erhard

# APC 10x7 Thin Electric Propeller Geometry from UIUC Propeller Database.


import SUAVE
from SUAVE.Core import Data, Units
import numpy as np
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_polars import (
    compute_airfoil_polars,
)
import os



def propeller_geometry():

    # --------------------------------------------------------------------------------------------------
    # Propeller Geometry:
    # --------------------------------------------------------------------------------------------------

    prop = SUAVE.Components.Energy.Converters.Propeller()
    
    prop.tag              = 'apc_10x7_propeller'
    prop.tip_radius       = 5 * Units.inches
    prop.number_of_blades = 2
    prop.hub_radius       = prop.tip_radius * 0.1
    prop.inputs.omega     = np.array([[4500 * Units.rpm]])    
    

    r_R = np.array(
        [
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
        ]
    )
    c_R = np.array(
        [
            0.138,
            0.154,
            0.175,
            0.190,
            0.198,
            0.202,
            0.200,
            0.195,
            0.186,
            0.174,
            0.161,
            0.145,
            0.129,
            0.112,
            0.096,
            0.081,
            0.061,
        ]
    )
    beta = np.array(
        [
            37.86,
            45.82,
            44.19,
            38.35,
            33.64,
            29.90,
            27.02,
            24.67,
            22.62,
            20.88,
            19.36,
            17.98,
            16.74,
            15.79,
            14.64,
            13.86,
            12.72,
        ]
    )

    prop.pitch_command       = 0.0* Units.deg
    prop.twist_distribution  = beta * Units.deg
    prop.chord_distribution  = c_R * prop.tip_radius
    prop.radius_distribution = r_R * prop.tip_radius
    prop.max_thickness_distribution = 0.12
    
    prop.number_azimuthal_stations = 24
    prop.number_radial_stations    = len(r_R)
    
    # Distance from mid chord to the line axis out of the center of the blade - In this case the 1/4 chords are all aligned 
    MCA = prop.chord_distribution/4. - prop.chord_distribution[0]/4.  
    prop.mid_chord_alignment = MCA
    
    airfoils_path = os.path.join(os.path.dirname(__file__), "../Airfoils/")
    polars_path = os.path.join(os.path.dirname(__file__), "../Airfoils/Polars/")
    airfoil_geometry = [airfoils_path + "Clark_y.txt"]
    airfoil_polars = [
        [   polars_path + "Clark_y_Ma_0.0_Re_0.05e6.txt",
            polars_path + "Clark_y_Ma_0.0_Re_0.1e6.txt", 
            polars_path + "Clark_y_Ma_0.0_Re_0.2e6.txt", 
            polars_path + "Clark_y_Ma_0.0_Re_0.5e6.txt", 
            polars_path + "Clark_y_Ma_0.0_Re_1.0e6.txt", 
            polars_path + "Clark_y_Ma_0.0_Re_2.0e6.txt", 
            polars_path + "Clark_y_Ma_0.0_Re_5.0e6.txt", 
        ],
    ]
    prop.airfoil_data = compute_airfoil_polars(airfoil_geometry, airfoil_polars)
    prop.airfoil_polar_stations = np.zeros(len(r_R))
    prop.airfoil_polar_stations = list(prop.airfoil_polar_stations.astype(int))
    
    return prop
