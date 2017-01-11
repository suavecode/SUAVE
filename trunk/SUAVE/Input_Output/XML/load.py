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

def load(file_in):
    """ A simple function to converts XML data into native Python object. """
    
    # open file, read conents
    if isinstance(file_in,str):
        file_in = open(file_in)
    src = file_in.read()
    
    # build xml tree
    builder = TreeBuilder()
    if isinstance(src,basestring):
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

class TreeBuilder(xml.sax.handler.ContentHandler):
    
    def __init__(self):

        self.root = XML_Data()
        
        self.stack = []
        self.current = self.root
        self.text_parts = []
        
    def startElement(self, name, attrs):
        
        self.stack.append((self.current, self.text_parts))
        
        self.current = XML_Data()
        self.current.tag = name
        
        self.text_parts = []
        
        for k, v in attrs.items():
            self.current.attributes[k] = v
            
    def endElement(self, name):
        
        text = ''.join(self.text_parts).strip()
        
        self.current.content = text
        element = self.current

        self.current, self.text_parts = self.stack.pop()
        
        self.current.elements.append(element)
        
    def characters(self, content):
        
        self.text_parts.append(content)

        

