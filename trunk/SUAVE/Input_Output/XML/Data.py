# Data.py
#
# Created: T. Lukaczyk Feb 2015
# Updated:  

""" SUAVE Methods for IO """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data as Data_Base

# for enforcing attribute style access names
import string
chars = string.punctuation + string.whitespace
t_table = string.maketrans( chars          + string.uppercase , 
                            '_'*len(chars) + string.lowercase )

# ----------------------------------------------------------------------
#  XML Data Clas
# ----------------------------------------------------------------------

class Data(Data_Base):
    
    def __defaults__(self):
        self.tag        = ''
        self.attributes = Attributes()
        self.content    = ''
        self.elements   = []
        
    def get_elements(self,tag):
        output = []
        for e in self.elements:
            if e.tag == tag:
                output.append(e)
        return output
    
    def new_element(self,tag):
        elem = Data()
        elem.tag = tag
        self.elements.append(elem)
        return elem
    
    @staticmethod
    def from_dict(data):
        
        result = Data()
        
        if data.has_key('tag'):
            result.tag = data.tag.translate(t_table)
        else:
            result.tag = 'node'
        
        for key,value in data.items():
            if isinstance( value, dict ):
                element = Data.from_dict(value)
                element.tag = key
                result.elements.append(element)
            else:
                element = Data()
                element.tag = key
                element.content = str(value)
                result.elements.append(element)
                
        return result 
         
    def __str__(self,indent=''):
        
        args = ''
        new_indent = '  '
        
        if not indent:
            args += self.dataname()  + '\n'        
        
        # tag
        args += indent + 'tag : %s\n' % self.tag
        
        # attributes
        if self.attributes:
            args += indent + 'attributes : %s' % self.attributes.__str__(indent+new_indent)
        else:
            args += indent + 'attributes : {}\n'
        
        # content
        args += indent + 'content : %s\n' % self.content
        
        # elements
        args += indent + 'elements : '

        # empty elements
        if not self.elements:
            args += '[]\n'
            
        # not empty elements
        else:
            args += '\n'
            indent += new_indent
            
            for i, e in enumerate(self.elements):
                args += indent + '%i :\n' % i
                args += e.__str__(indent + new_indent)
            
        return args
            
            
# ----------------------------------------------------------------------
#  XML Attributes Clas
# ----------------------------------------------------------------------
class Attributes(Data_Base):
    pass