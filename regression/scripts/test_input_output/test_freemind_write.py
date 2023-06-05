# test_freemind.py
# 
# Created:  Trent Lukaczyk, Feb 2015
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  

import SUAVE.Input_Output as IO
import SUAVE.Core.Data as Data


# ----------------------------------------------------------------------        
#   The Test
# ----------------------------------------------------------------------  

def main():
    output = Data()
    
    output.hello = Data()
    output.hello.world = 'watsup'
    output.hello.now = None
    output.hello['we are'] = None
    output.hello.rockin = ['a','list','of','items']
    
    IO.FreeMind.save(output,'output.mm')


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------  
    
if __name__ == '__main__':
    main()