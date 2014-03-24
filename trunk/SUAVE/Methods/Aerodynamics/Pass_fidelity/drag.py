


#-----------------------------over aircraft drag---------------------------------------------       
#def drag(l_fus, d_fus,l_nose,l_tail,mac_w,t_c_w,sweep_w, S_exposed_w, mac_h,t_c_h,sweep_h, S_exposed_h,mac_v,t_c_v,sweep_v, S_exposed_v,d_engexit,Sref,Mc,roc,muc ,Tc,Cl, AR, e,S_affected_w,S_affected_h,S_affected_v ):
#def drag(self,fus_l, d_fus,l_nose,l_tail,mac_w,t_c_w,sweep_w, S_exposed_w, mac_h,t_c_h,sweep_h, S_exposed_h,mac_v,t_c_v,sweep_v, S_exposed_v,d_engexit,Sref,Mc,rhoc,muc ,Tc,Cl, AR, e ):
#def drag(vehicle,Segment):
def drag(vehicle,state,cl_w,cd_w,curr_itr):
    ''' outputs = method1(inputs)
           cd_p inputs  geometry related : fuselage  -  fus_l, d_fus,l_nose,l_tail 
                            wing      -  mac_w,t_c_w,sweep_w, S_exposed_w
                            horiz stab-  mac_h,t_c_h,sweep_h, S_exposed_h
                            vert stab -  mac_v,t_c_v,sweep_v, S_exposed_v
                            eng nac   -  d_engexit
                            general   -  Sref

         mission related  : Mc,Tinf,rhoc,muc ,Tc  
         constants        : gamma, R,    more documentation


           cd_i  inputs    :  Cl, AR, e
            more documentation

    '''
    
    #unpacking
    
       
    Wing1=state.config.Wings[0]
    Wing2=state.config.Wings[1]
    Wing3=state.config.Wings[2]                                        
    
    
    l_fus=state.config.Fuselages[0].length_cabin
    d_fus=state.config.Fuselages[0].width
    l_nose=state.config.Fuselages[0].length_nose
    l_tail=state.config.Fuselages[0].length_tail
    
    
    mac_w=Wing1.chord_mac
    t_c_w=Wing1.t_c
    sweep_w=Wing1.sweep
    S_exposed_w=Wing1.S_exposed
    S_affected_w=Wing1.S_affected
    arw_w=Wing1.ar
    span_w=Wing1.span
    
    mac_h=Wing2.chord_mac
    t_c_h=Wing2.t_c
    sweep_h=Wing2.sweep
    S_exposed_h=Wing2.S_exposed
    S_affected_h=Wing2.S_affected
    arw_h=Wing2.ar
    span_h=Wing2.span
    
    mac_v=Wing3.chord_mac
    t_c_v=Wing3.t_c
    sweep_v=Wing3.sweep
    S_exposed_v=Wing3.S_exposed
    S_affected_v=Wing3.S_affected
    arw_v=Wing3.ar
    span_v=Wing3.span
    
    d_engexit=state.config.Propulsors[0].df
    Sref=Wing1.sref
    #Mc=Segment.mach
    Mc=state.M[curr_itr]
        
    #--obtained from atmos    
    roc=state.rho[curr_itr]
    muc =state.mew[curr_itr]
    Tc=state.T[curr_itr]
    
   #obtained from aero itself wait for lift model 
    #Cl_w=Wing1.Cl
    #Cl_h=Wing2.Cl
    #Cl_v=Wing3.Cl    
    
    
    e_w=Wing1.e
    e_h=Wing2.e
    e_v=Wing3.e    
    
    Cl_w=cl_w[0]
    Cl_h=cl_w[1]
    Cl_v=cl_w[2] 
    
    
    
    cdi_w=cd_w[0]
    cdi_h=cd_w[1]
    cdi_v=cd_w[2]
    
    
    
    
    #----calling necessary functions for drag computation 
    
        
    [cd_p,cdp_w,cdp_v,cdp_h]=cdp(l_fus, d_fus,l_nose,l_tail,mac_w,t_c_w,sweep_w, S_exposed_w, mac_h,t_c_h,sweep_h, S_exposed_h,mac_v,t_c_v,sweep_v, S_exposed_v,d_engexit,Sref,Mc,roc,muc ,Tc,S_affected_w,S_affected_h,S_affected_v)
    cd_i_w=cdi(Cl_w, arw_w, cdi_w,cdp_w,d_fus/span_w)
    cd_i_h=cdi(Cl_h, arw_h, cdi_h,cdp_v,d_fus/span_h)
    cd_i_v=cdi(Cl_v, arw_v, cdi_v,cdp_h,d_fus/span_v)
    
    cd_i= cd_i_w + cd_i_h + cd_i_v
    
    cd_c=cdc(Mc,sweep_w,t_c_w,Cl_w)

    cd_tot=cd_p+cd_i+cd_c   #without trim
    
    cd_tot_w_trim=1.02*cd_tot #with trim
    
    

    #code        
    #outputs = 'total aircraft drag coeff'

    return cd_tot
