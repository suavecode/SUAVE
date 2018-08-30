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
def expand_sub_segments(segment):
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
    
    pass

    last_tag = None
    
<<<<<<< HEAD
    for tag,sub_segment in list(segment.segments.items()):

=======
    for tag,sub_segment in segment.segments.items():
        
>>>>>>> feature-P3
        if Process.verbose:
            print('segment start :' , tag)
        
        if last_tag:
            sub_segment.state.initials = segment.segments[last_tag].state
        last_tag = tag        

               
        if Process.verbose:
            print('segment end :' , tag)        


# ----------------------------------------------------------------------
#  Update Sub Segments
# ----------------------------------------------------------------------        

## @ingroup Methods-Missions-Segments-Common
def update_sub_segments(segment):
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
    
<<<<<<< HEAD
    for tag,sub_segment in list(segment.segments.items()):
        sub_segment.initialize()
        sub_segment.iterate()
        sub_segment.finalize()
=======
    for tag,sub_segment in segment.segments.items():
        sub_segment.initialize(state.segments[tag])
        sub_segment.iterate(state.segments[tag])
        sub_segment.finalize(state.segments[tag])
>>>>>>> feature-P3
                         
# ----------------------------------------------------------------------
#  Finalize Sub Segments
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def finalize_sub_segments(segment):
    """ Sets the conditions in each sub segment for a mission
    
        Assumptions:
        N/A
        
        Inputs:
        N/A
            
        Outputs:
        N/A

        Properties Used:
        N/A
                                
<<<<<<< HEAD
    """           

    for tag,sub_segment in list(segment.segments.items()):
        sub_segment.finalize()

=======
    """       
    
    from SUAVE.Analyses.Mission.Segments.Conditions import Conditions
    
    for tag,sub_segment in segment.segments.items():
        sub_segment.finalize(state.segments[tag])
        state.segments[tag].initials = Conditions()
>>>>>>> feature-P3

# ----------------------------------------------------------------------
#  Sequential Sub Segments
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def sequential_sub_segments(segment):
    
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
<<<<<<< HEAD

    for tag,sub_segment in list(segment.segments.items()):
        sub_segment.evaluate()
=======
    
    
    for tag,sub_segment in segment.segments.items():
        sub_segment.evaluate(state.segments[tag])
>>>>>>> feature-P3
