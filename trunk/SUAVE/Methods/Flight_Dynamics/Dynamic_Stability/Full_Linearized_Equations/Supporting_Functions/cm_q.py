# cm_q.py
# 
# Created:  Jun 2014, A. Wendorff
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

def cm_q(cm_i, l_t, mac):
    # Generating Stability derivative
    
    cm_q = 2. * 1.1 * cm_i * l_t / mac # Check function
    
    return cm_q