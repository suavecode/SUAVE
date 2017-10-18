## @ingroup Optimization-Package_Setups
# dummy_mission_solver.py
#
# Created:  Oct 2017, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#  Dummy mission solver
# ----------------------------------------------------------------------

def dummy_mission_solver( iterate, unknowns, args=(), xtol = 0., full_output=1):
    """Rather than run a mission solver this solves a mission at a particular instance.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    iterate      [suave function]
    unknowns     [inputs to the function]
    args         [[segment,state]]
    xtol         [irrelevant]
    full_output  [irrelevant]

    Outputs:
    unknowns     [inputs to the function]
    infodict     [None]
    ier          [1]
    msg          [string]

    Properties Used:
    N/A
    """       
    
    
    iterate(unknowns,args)
    
    infodict = None
    ier      = 1
    msg      = 'Used dummy mission solver'
    
    return unknowns, infodict, ier, msg