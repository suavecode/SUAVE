## @ingroup Input_Output-XML
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
t_table = str.maketrans( chars          + string.ascii_uppercase , 
                            '_'*len(chars) + string.ascii_lowercase )

# ----------------------------------------------------------------------
#  XML Data Clas
# ----------------------------------------------------------------------
## @ingroup Input_Output-XML
class Data(Data_Base):
    """This the XML data class used in SUAVE.

    Assumptions:
    None

    Source:
    N/A
    """       
    def __defaults__(self):
        """Defaults for the data class.
    
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
        self.tag        = ''
        self.attributes = Attributes()
        self.content    = ''
        self.elements   = []
        
    def get_elements(self,tag):
        """Gets elements with a given tag.
    
        Assumptions:
        None
    
        Source:
        N/A
    
        Inputs:
        tag     - used to check which elements to return
    
        Outputs:
        output  - list of matching elements
    
        Properties Used:
        N/A
        """           
        output = []
        for e in self.elements:
            if e.tag == tag:
                output.append(e)
        return output
    
    def new_element(self,tag):
        """Creates a new element.
    
        Assumptions:
        None
    
        Source:
        N/A
    
        Inputs:
        tag   - tag of the new element
    
        Outputs:
        elem  - the new element
    
        Properties Used:
        N/A
        """           
        elem = Data()
        elem.tag = tag
        self.elements.append(elem)
        return elem
    
    @staticmethod
    def from_dict(data):
        """Gives a list of elements from data.
    
        Assumptions:
        None
    
        Source:
        N/A
    
        Inputs:
        data    - data class to read
    
        Outputs:
        results - list of elements
    
        Properties Used:
        N/A
        """           
        result = Data()
        
        if 'tag' in data:
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
        """Determines how the class is shown in a string.
    
        Assumptions:
        None
    
        Source:
        N/A
    
        Inputs:
        indent  <string> indent to be used
    
        Outputs:
        args    <string>
    
        Properties Used:
        N/A
        """           
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
## @ingroup Input_Output-XML
class Attributes(Data_Base):
    """Placeholder class. No functionality.

    Assumptions:
    None

    Source:
    N/A
    """       
    pass