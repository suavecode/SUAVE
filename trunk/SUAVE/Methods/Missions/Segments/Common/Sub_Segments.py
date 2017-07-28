## @ingroup Methods-Missions-Segments-Common
# Sub_Segments.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero
#           Mar 2016, E. Botero
#           Jul 2017, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from copy import deepcopy
from SUAVE.Analyses import Process

# ----------------------------------------------------------------------
#  Expand Sub Segments
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common   
def expand_sub_segments(segment,state):
    """ Fills in the segments to a mission with data, sets initials data if necessary
    
        Assumptions:
        N/A
        
        Inputs:
        N/A
            
        Outputs:
        N/A

        Properties Used:
        N/A
                                
    """    

    last_tag = None
    
    for tag,sub_segment in segment.segments.items():
        
        if Process.verbose:
            print 'segment start :' , tag
        
        sub_state = deepcopy( sub_segment.state )
        
        if last_tag:
            sub_state.initials = state.segments[last_tag]
        last_tag = tag        
        
        sub_segment.initialize(sub_state)
        
        state.segments[tag]     = sub_state
        state.unknowns[tag]     = sub_state.unknowns
        state.conditions[tag]   = sub_state.conditions
        state.residuals[tag]    = sub_state.residuals
        
        if Process.verbose:
            print 'segment end :' , tag        


# ----------------------------------------------------------------------
#  Update Sub Segments
# ----------------------------------------------------------------------        

## @ingroup Methods-Missions-Segments-Common
def update_sub_segments(segment,state):
    """ Loops through the segments and fills them in
    
        Assumptions:
        N/A
        
        Inputs:
        N/A
            
        Outputs:
        N/A

        Properties Used:
        N/A
                                
    """      
    
    for tag,sub_segment in segment.segments.items():
        sub_segment.initialize(state.segments[tag])
        sub_segment.iterate(state.segments[tag])
        sub_segment.finalize(state.segments[tag])
                         
# ----------------------------------------------------------------------
#  Finalize Sub Segments
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def finalize_sub_segments(segment,state):
    """ Sets the conditions in each sub segment for a mission
    
        Assumptions:
        N/A
        
        Inputs:
        N/A
            
        Outputs:
        N/A

        Properties Used:
        N/A
                                
    """       
    
    from SUAVE.Analyses.Mission.Segments.Conditions import Conditions
    
    for tag,sub_segment in segment.segments.items():
        sub_segment.finalize(state.segments[tag])
        state.segments[tag].initials = Conditions()

# ----------------------------------------------------------------------
#  Sequential Sub Segments
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def sequential_sub_segments(segment,state):
    
    """ Evaluates all the segments in a mission one by one
    
        Assumptions:
        N/A
        
        Inputs:
        N/A
            
        Outputs:
        N/A

        Properties Used:
        N/A
                                
    """       
    
    
    for tag,sub_segment in segment.segments.items():
        sub_segment.evaluate(state.segments[tag])