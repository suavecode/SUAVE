# save_tree.py
#
# Created: T. Lukaczyk Feb 2015
# Updated: Carlos Ilario, Feb 2016 

""" SUAVE Methods for IO """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Tree_Element import Tree_Element
import json


# ----------------------------------------------------------------------
#  The Method
# ----------------------------------------------------------------------

def save_tree(data,filename,root_name=None):

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

def to_d3(tree,data):
    
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