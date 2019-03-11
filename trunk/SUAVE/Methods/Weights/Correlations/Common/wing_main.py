## @ingroup Methods-Weights-Correlations-Common 
# wing_main.py
#
# Created:  Jan 2014, A. Wendorff
# Modified: Feb 2014, A. Wendorff
#           Feb 2016, E. Botero
#           Jul 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units
import numpy as np

# ----------------------------------------------------------------------
#   Wing Main
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Common 
def wing_main(wing,Nult,TOW,wt_zf,rho,sigma):
    """ Calculate the wing weight of the aircraft based on the fully-stressed 
    bending weight of the wing box
    
    Assumptions:
        calculated total wing weight based on a bending index and actual data 
        from 15 transport aircraft 
    
    Source: 
        http://aerodesign.stanford.edu/aircraftdesign/AircraftDesign.html
        search for: Derivation of the Wing Weight Index
        
    Inputs:
        wing
             .span.projected                         [meters**2]
             .taper                                  [dimensionless]
             .sweeps.quarter_chord                   [radians]
             .thickness_to_chord                     [dimensionless]
             .Segments
                 .root_chord_percent                 [dimensionless]
                 .thickness_to_chord                 [dimensionless]
                 .percent_span_location              [dimensionless]

        Nult - ultimate load factor of the aircraft  [dimensionless]
        TOW - maximum takeoff weight of the aircraft [kilograms]
        wt_zf - zero fuel weight of the aircraft     [kilograms]
    
    Outputs:
        weight - weight of the wing                  [kilograms]          
        
    Properties Used:
        N/A
    """ 
    
    # unpack inputs
    span  = wing.spans.projected# / Units.ft
    taper = wing.taper
    sweep = wing.sweeps.quarter_chord
    area  = wing.areas.reference #/ Units.ft**2 
    t_c_w = wing.thickness_to_chord
    RC    = wing.chords.root# / Units.feet
     
    mtow  = TOW #/ Units.lb # Convert kg to lbs
    zfw   = wt_zf# / Units.lb # Convert kg to lbs            
    
    rho_sigma = rho*9.81/sigma
    #rho_sigma = (1.642*10.**-6.)
    
    # Start the calculations
    l_tot = Nult*np.sqrt(mtow*zfw)*9.81
    gamma = 16*l_tot*rho_sigma/(np.pi*span)
    L0 = 2*l_tot/(span*np.pi)
                      
    if len(wing.Segments)>0:
        
        # Prime some numbers
        run_sum = 0
        run_sum2 = 0
        run_sum3 = 0
        b = span
        
        for i in range(1,len(wing.Segments)):
            
            # Unpack segment level info            
            Y1 = wing.Segments[i-1].percent_span_location
            Y2 = wing.Segments[i].percent_span_location
            
            if wing.Segments[i-1].root_chord_percent==wing.Segments[i].root_chord_percent and wing.Segments[i-1].thickness_to_chord==wing.Segments[i].thickness_to_chord:
                C  = wing.Segments[i-1].root_chord_percent*RC
                G  = wing.Segments[i].thickness_to_chord
            
                WB = (1/(G*C)) * 1/3*(1/8*(-Y1*(5-2*Y1**2)*np.sqrt(1-Y1**2)-3*np.arcsin(Y1))+1/8*(Y2*(5-2*Y2**2)*np.sqrt(1-Y2**2)+3*np.arcsin(Y2)))
            
            else:
                # A is the root thickness
                A = RC*wing.Segments[i-1].root_chord_percent* wing.Segments[i-1].thickness_to_chord
                # B is the slope of the thickness
                B = (A-RC*wing.Segments[i].root_chord_percent* wing.Segments[i].thickness_to_chord)/(wing.Segments[i].percent_span_location - wing.Segments[i-1].percent_span_location)
                # C is the offset
                C = wing.Segments[i-1].percent_span_location
                
                WB1 = big_integral(Y1, A, B, C)
                WB2 = big_integral(Y2, A, B, C)
                
                WB  = WB2-WB1
                
                
            run_sum3+= np.real(WB)
            
        weight_factor4 = rho_sigma*(b**2)*L0*run_sum3/2
        
        weight = 4.22*area / Units.feet**2 + (weight_factor4 / Units.lb)
            
    else:
        
        area  = wing.areas.reference / Units.ft**2 
        span  = wing.spans.projected  / Units.ft 
        mtow  = TOW / Units.lb # Convert kg to lbs
        zfw   = wt_zf / Units.lb # Convert kg to lbs        

        #Calculate weight of wing for traditional aircraft wing
        weight = 4.22*area + 1.642*10.**-6. * Nult*(span)**3. *(mtow*zfw)**0.5 \
                 * (1.+2.*taper)/(t_c_w*(np.cos(sweep))**2. * area*(1.+taper) )
 
    weight = weight * Units.lb # Convert lb to kg

    return weight


def big_integral(x,A,B,C):
    
    results = (1/(4*B**3))*(2*A**2*(np.pi)*x+4*A*B*C*(np.pi)*x+ \
            B**2*(-1+2*C**2)*(np.pi)*x- \
            2*(2*A**2+4*A*B*C+B**2*(-1+2*C**2))*np.sqrt(1-x**2)+ \
            2/3*B*np.sqrt(1-x**2)*(3*A*x+B*(-1+3*C*x+x**2))+ \
            2*B*(A+B*C)*np.arcsin(x)- \
            2*(2*A**2+4*A*B*C+B**2*(-1+2*C**2))*x*np.arcsin(x)-( \
            4*(A+B*C)**2*np.sqrt(0j+-A**2-2*A*B*C-B**2*(-1+C**2))* \
            np.log(A+B*C-B*x))/B-( \
            4*(A**3+3*A**2*B*C+B**3*C*(-1+C**2)+A*B**2*(-1+3*C**2))*x*np.log( \
            A+B*C-B*x))/np.sqrt(0j+-A**2-2*A*B*C-B**2*(-1+C**2))+( \
            8*(A+B*C)**2*np.sqrt(0j-A**2-2*A*B*C-B**2*(-1+C**2))* \
            np.log(0j-A-B*C+B*x))/ \
            B+(4*(A**3+3*A**2*B*C+B**3*C*(-1+C**2)+ \
            A*B**2*(-1+3*C**2))*x*np.log(0j-B+A*x+B*C*x- \
            np.sqrt(0j+-A**2+B**2-2*A*B*C-B**2*C**2)*np.sqrt( \
            1-x**2)))/(np.sqrt(0j-A**2-2*A*B*C-B**2*(-1+C**2)))-(1/B)* \
            4*(A+B*C)**2*np.sqrt(0j-A**2-2*A*B*C-B**2*(-1+C**2))* \
            np.log(-B+A*x+B*C*x+ \
            np.sqrt(0j-A**2+B**2-2*A*B*C-B**2*C**2)*np.sqrt(1-x**2))+(1/B)* 4*(A+B*C)*np.sqrt(0j-(A**2+2*A*B*C+B**2*(-1+C**2))**2)*np.log(A**2*x+2*A*B*C*x+B**2*(-1+C**2)*x+ \
            np.sqrt(0j-(A**2+2*A*B*C+B**2*(-1+C**2))**2)*np.sqrt(1-x**2)))
                
    return results