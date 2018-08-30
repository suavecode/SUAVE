## @ingroup Input_Output-D3JS
# save_tree.py
#
# Created: T. Lukaczyk Feb 2015
# Updated: Carlos Ilario, Feb 2016 

""" SUAVE Methods for IO """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Tree_Element import Tree_Element
import json


# ----------------------------------------------------------------------
#  The Method
# ----------------------------------------------------------------------
## @ingroup Input_Output-D3JS
def save_tree(data,filename,root_name=None):
    """This creates a D3JS file based on a SUAVE data structure.
    
    Assumptions:
    None

    Source:
    N/A

    Inputs:
    data       SUAVE data structure
    filename   <string> name of the output file
    root_name  (optional) root name for the data structure in the output, default is data.tag

    Outputs:
    D3JS file with name as specified by filename

    Properties Used:
    N/A
    """       

    if not isinstance(data,Tree_Element):
        if 'tag' in data:
            root_name = data.tag
        elif root_name:
            root_name = root_name
        else:
            root_name = 'output'
            
        tree = Tree_Element(root_name)
        
        # translate
        to_d3(tree,data)
        
    else:
        tree = data
    
    with open(filename,'w') as output:
        json.dump(tree, output, indent=2)    

        
# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------
## @ingroup Input_Output-D3JS
def to_d3(tree,data):
    """This puts data for a SUAVE data structure into the needed format for D3JS
    
    Assumptions:
    None

    Source:
    N/A

    Inputs:
    tree     upper level data structure
    data     SUAVE data structure

    Outputs:
    None - modifies tree

    Properties Used:
    N/A
    """      
    
    tree.children = []
    
    for k,v in data.items():
        
        e = Tree_Element(k)
        tree.children.append(e)
        
        if isinstance(v,dict):
            to_d3(e,v)
            
        else:
            v = Tree_Element( str(v) )
            e.children = []
            e.children.append(v)
   
    if not tree.children:
        tree.children.append( Tree_Element('{}') )