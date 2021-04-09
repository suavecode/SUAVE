# test_xml.py
# 
# Created:  Trent Lukaczyk, Feb 2015
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  

import filecmp
import SUAVE

# ----------------------------------------------------------------------        
#   The Test
# ----------------------------------------------------------------------  

def main():
    
    data = SUAVE.Input_Output.XML.load('example.xml')
    
    print(data)
    
    SUAVE.Input_Output.XML.save(data,'output.xml')
    
    result = filecmp.cmp('example.xml','output.xml')
    
    
# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------  
    
if __name__ == '__main__':
    main()