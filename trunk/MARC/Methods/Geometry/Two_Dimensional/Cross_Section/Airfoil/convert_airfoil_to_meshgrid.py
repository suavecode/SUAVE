## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# convert_airfoil_to_meshgrid.py
# 
# Created:    Dec 2022, J. Smart
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

from MARC.Core import Data
import numpy as np

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def convert_airfoil_to_meshgrid(airfoil_geometry, *args, **kwargs):
    """Converts a MARC airfoil geometry representation to a Numpy meshgrid
    array mask of boolean values.

    Assumptions:
    None

    Source:
    None

    Inputs:
    airfoil_geometry                [MARC Data Structure]
        .x_lower_surface            [Numpy Array, float32]
        .y_lower_surface            [Numpy Array, float32]
        .y_upper_surface            [Numpy Array, float32]

    Outputs: 

    airfoil_meshgrid                [Numpy Array, bool]

    Properties Used:
    N/A	
    """

    # Unpack Values

    x_lower_surface = airfoil_geometry.x_lower_surface
    y_lower_surface = airfoil_geometry.y_lower_surface
    y_upper_surface = airfoil_geometry.y_upper_surface

    # Determine necessary resolution of the meshgrid. We do this by dividing the
    # x-length of the  lower surface by the minimum separation between
    # any two x-coordinates of the geometry (ceil-rounded to an int). Later
    # we'll instantiate the meshgrid with this number of x-indices, so that the
    # separation between any two points in the meshgrid is equal to the minimum
    # separation between any two x-coordinates.

    x_length = (
        np.max(x_lower_surface)
    )

    Nx = np.ceil(
        x_length / np.abs(np.min(np.diff(x_lower_surface)))
    ).astype(int)

    # We determine the necessary number of y-coordinate points by taking the
    # maximum separation between the highest point of the upper surface and the
    # lowest point of the lower surface and multiplying that by the number of
    # x-points in order to re-normalize to our future meshgrid coordinates,
    # then ciel-rounding to an int.

    Ny = np.ceil(
        Nx * ( np.max(y_upper_surface) - np.min(y_lower_surface) )
    ).astype(int)

    # Instantiate the meshgrid, using ij-indexing so that X[i,j] returns i
    # for all points, and Y[i,j] returns j for all coordinates.

    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")

    # Create the indexing arrays for the meshgrid. These convert the airfoil
    # geometry coordinates into meshgrid array indices. The X_INDICES are found
    # just by multplying/stretching the x_lower_surface coordinates across the
    # number of x-coodinates in the meshgrid.

    X_INDICES = np.ceil(
        Nx / x_length * x_lower_surface
    ).astype(int)

    # The Y_INDICES are similarly stretched, but first are offset by the
    # minimum of the lower surface to bring them to a relative zero

    Y_LOWER_INDICES = np.floor(
        Nx / x_length * (
            y_lower_surface - np.min(y_lower_surface)
        )
    ).astype(int)

    Y_UPPER_INDICES = np.ceil(
        Nx /x_length * (
            y_upper_surface - np.min(y_lower_surface)
        )
    ).astype(int)

    # We then repeat the elements of the Y_INDICES by the number of gridpoints
    # between each x-coordinate, essentially treating the y-surface as flat
    # between those points. We trim the final point by telling it to repeat 0
    # times

    REPEATS = np.append(
            np.diff(X_INDICES),
            0
    )

    # Need to hand the case where the X_INDICES aren't sorted, and swap
    # some elements around to allow the masks to be created

    if np.any(REPEATS<0):

        REPEAT_FLAG = True

        NEG_REPEATS = np.where(REPEATS<0)[0]

        if np.any(np.diff(NEG_REPEATS) == 1):
            print("Airfoil geometry contains sequential negative x-steps. Meshing Failed.")
            return None

        (X_INDICES[NEG_REPEATS],
         X_INDICES[NEG_REPEATS + 1]) = (X_INDICES[NEG_REPEATS + 1],
                                        X_INDICES[NEG_REPEATS])

        (Y_LOWER_INDICES[NEG_REPEATS],
         Y_LOWER_INDICES[NEG_REPEATS + 1]) = (Y_LOWER_INDICES[NEG_REPEATS + 1],
                                              Y_LOWER_INDICES[NEG_REPEATS])

        (Y_UPPER_INDICES[NEG_REPEATS],
         Y_UPPER_INDICES[NEG_REPEATS + 1]) = (Y_UPPER_INDICES[NEG_REPEATS + 1],
                                              Y_UPPER_INDICES[NEG_REPEATS])

        REPEATS = np.append(
            np.diff(X_INDICES),
            0
        )

        Nx = np.sum(REPEATS)

        Ny = np.ceil(
            Nx * (np.max(y_upper_surface) - np.min(y_lower_surface))
        ).astype(int)

        X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")

    Y_LOWER_INDICES = np.repeat(Y_LOWER_INDICES, REPEATS)
    Y_UPPER_INDICES = np.repeat(Y_UPPER_INDICES, REPEATS)

    # We then create masks for the upper and lower surfaces by tiling the
    # indices over the meshgrid (taking a transpose to comport with our earlier
    # indexing style).

    Y_LOWER_GRID = np.tile(Y_LOWER_INDICES, (Ny,1)).T
    Y_UPPER_GRID = np.tile(Y_UPPER_INDICES, (Ny,1)).T

    # We then create our airfoil meshgrid mask by comparing our Y coordinates
    # from the meshgrid to our upper and lower grids, intermediately treating
    # them as ints to simplify the multi-condition comparison

    Y_LOWER = (Y > Y_LOWER_GRID).astype(int)
    Y_UPPER = (Y < Y_UPPER_GRID).astype(int)

    AIRFOIL_MASK = (Y_LOWER + Y_UPPER) > 1

    return AIRFOIL_MASK
