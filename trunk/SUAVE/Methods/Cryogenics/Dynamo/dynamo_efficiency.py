## @ingroup Methods-Cryogenic-Dynamo
# dynamo_efficiency.py
#
# Created:  Feb 2022,  S. Claridge

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE
import numpy as np
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Estimated efficiency of HTS Dynamo 
# ----------------------------------------------------------------------
## @ingroup Methods-Cryogenic-Dynamos
def efficiency_curve(Dynamo, current):

    """ This sets the default values.

    Assumptions:
        The efficiency curve of the Dynamo is a parabola 

    Source:
        "Practical Estimation of HTS Dynamo Losses" - Kent Hamilton, Member, IEEE, Ratu Mataira-Cole, Jianzhao Geng, Chris Bumby, Dale Carnegie, and Rod Badcock, Senior Member, IEEE

    Inputs:
        current        [A]

    Outputs:
        efficiency      [W/W]

    Properties Used:
        None
    """     

    x = np.array(current)

    if np.any(x > Dynamo.rated_current * 1.8 ) or np.any(x < Dynamo.rated_current * 0.2): #Plus minus 80
        print("Current out of range")
        return 0 

    a          = ( Dynamo.efficiency ) / np.square(Dynamo.rated_current) #one point on the graph is assumed to be  (0, 2 * current), 0  = a (current ^ 2) + efficiency 
    
    efficiency = -a * (np.square( x - Dynamo.rated_current) ) +  Dynamo.efficiency # y = -a(x - current)^2 + efficieny 

    return   efficiency