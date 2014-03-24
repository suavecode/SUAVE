# test_loads.py

import SUAVE
import test_b737_pass

# MAIN
def main():
    
    test()
    
    return
    

# TEST LOADS
def test():
    
    Vehicle = test_b737_pass.create_av()
    
    print Vehicle

    return


# call main
if __name__ == '__main__':
    main()
