

#-------------flap deflection------------------------------
def flap_lift(t_c,fc,fa,sweep,Sref,Swf):  # doc string defining variables

    #t_c=   0.1
    #fc=20   #flap chord as a % of wing chord
    #fa=20   #flap angle degrees
    Swf=0.7*Sref
    tc_r=t_c/100
    
    dmax_ref= -4E-05*tc_r**4 + 0.0014*tc_r**3 - 0.0093*tc_r**2 + 0.0436*tc_r + 0.9734
    #For single slotted flap multiply this value by 0.93. For triple slotted flaps, multiply by 1.08.
    
    
    K1=0.0395*fc + 0.0057
    K2=-0.0002*fa**2 + 0.0284*fa  + 0.0012
    
    
    K = (1-0.08*(numpy.cos(sweep))**2)*(numpy.cos(sweep))**0.75
    dmax_flaps = K1*K2*dmax_ref
    dcl_max_flaps= Swf/Sref*dmax_flaps*K
    return dcl_max_flaps

