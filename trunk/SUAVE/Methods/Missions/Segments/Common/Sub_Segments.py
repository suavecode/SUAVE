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

from SUAVE.Analyses import Process
from SUAVE.Core import Data

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

    last_tag = None
    
    for tag,sub_segment in segment.segments.items():
        
        if Process.verbose:
            print('segment start :' , tag)
        
        if last_tag:
            sub_segment.state.initials = segment.segments[last_tag].state
        last_tag = tag        
        
        sub_segment.process.initialize.expand_state(sub_segment)
               
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
    for tag,sub_segment in segment.segments.items():
        sub_segment.initialize()
        sub_segment.iterate()
        sub_segment.finalize()
   
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
                                
    """           

    for tag,sub_segment in segment.segments.items():
        sub_segment.finalize()


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

    for tag,sub_segment in segment.segments.items():
        sub_segment.evaluate()


# ----------------------------------------------------------------------
#  Sequential Sub Segments
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def merge_sub_segment_states(segment):
    
    """ Merges all of the sub segment states back into the main state
    
        Assumptions:
        N/A
        
        Inputs:
        N/A
            
        Outputs:
        N/A

        Properties Used:
        N/A
                                
    """       

    segment.state.update(segment.merged())

# ----------------------------------------------------------------------
#  Sequential Sub Segments
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def unpack_subsegments(segment):
    
    """ Unpacks the unknowns from the higher level state into the sub segments
    
        Assumptions:
        The subsegments have numerics
        
        Inputs:
        N/A
            
        Outputs:
        N/A

        Properties Used:
        N/A
                                
    """       

    # Build a dict with the sections, sections start at 0
    counter = Data()
    
    for key in segment.state.unknowns.keys():
        counter[key] = 0

    for i, sub_segment in enumerate(segment.segments):
        ctrl_pnts = sub_segment.state.numerics.number_control_points
        for key in sub_segment.state.unknowns.keys():
            sub_segment.state.unknowns[key] = segment.state.unknowns[key][counter[key]:counter[key]+ctrl_pnts]
            counter[key] = counter[key]+ctrl_pnts
            
    return
            
            