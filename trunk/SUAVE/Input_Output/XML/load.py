## @ingroup Input_Output-XML
# load.py
#
# Created: T. Lukaczyk Feb 2015
# Updated:  

""" SUAVE Methods for IO """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import xml.sax.handler

from .Data import Data as XML_Data


# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
## @ingroup Input_Output-XML
def load(file_in):
    """Converts an XML file into an XML data structure.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    file_in  - The XML file

    Outputs:
    xml_data - The XML data structure

    Properties Used:
    N/A
    """     
    
    # open file, read conents
    if isinstance(file_in,str):
        file_in = open(file_in)
    src = file_in.read()
    
    # build xml tree
    builder = TreeBuilder()
    if isinstance(src,str):
        xml.sax.parseString(src, builder)
    else:
        xml.sax.parse(src, builder)
    
    # close file
    file_in.close()

    # pull xml data
    xml_data = builder.root.elements[0]
    
    return xml_data


# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------
## @ingroup Input_Output-XML
class TreeBuilder(xml.sax.handler.ContentHandler):
    """A class used to build the tree in an XML data structure

    Assumptions:
    None

    Source:
    N/A
    """       
    
    def __init__(self):
        """Base values for the class to function.
    
        Assumptions:
        None
    
        Source:
        N/A
    
        Inputs:
        None
    
        Outputs:
        None
    
        Properties Used:
        N/A
        """   
        self.root = XML_Data()
        
        self.stack = []
        self.current = self.root
        self.text_parts = []
        
    def startElement(self, name, attrs):
        """Starts a new element. This is used by an external package.
    
        Assumptions:
        None
    
        Source:
        N/A
    
        Inputs:
        name    - tag for the data structure
        attrs   - items to be added to the data structure
    
        Outputs:
        None
    
        Properties Used:
        self.
          stack
          current.
            tag
            attributes
          text_parts
        """           
        self.stack.append((self.current, self.text_parts))
        
        self.current = XML_Data()
        self.current.tag = name
        
        self.text_parts = []
        
        for k, v in attrs.items():
            self.current.attributes[k] = v
            
    def endElement(self, name):
        """End a new element. This is used by an external package.
    
        Assumptions:
        None
    
        Source:
        N/A
    
        Inputs:
        None used
    
        Outputs:
        None
    
        Properties Used:
        self.
          stack
          current.
            content
            elements
          text_parts
        """               
        text = ''.join(self.text_parts).strip()
        
        self.current.content = text
        element = self.current

        self.current, self.text_parts = self.stack.pop()
        
        self.current.elements.append(element)
        
    def characters(self, content):
        """Appends content. This is used by an external package.
    
        Assumptions:
        None
    
        Source:
        N/A
    
        Inputs:
        content
    
        Outputs:
        None
    
        Properties Used:
        self.text_parts
        """           
        self.text_parts.append(content)

        

