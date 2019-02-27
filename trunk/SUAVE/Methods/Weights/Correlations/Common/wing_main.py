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
    span  = wing.spans.projected
    taper = wing.taper
    sweep = wing.sweeps.quarter_chord
    area  = wing.areas.reference
    t_c_w = wing.thickness_to_chord
    
    rho_sigma = rho/sigma
    #rho_sigma = (1.642*10.**-6.)
    
    
    # Start the calculations
    l_tot = Nult*np.sqrt(TOW*wt_zf )*9.81
    gamma = 16*l_tot*rho_sigma/(np.pi*span)    
    
    if len(wing.Segments)>0:

        
        # Prime some numbers
        RC = wing.chords.root
        run_sum = 0
        b = span
        
        for i in range(1,len(wing.Segments)):
            C  = wing.Segments[i-1].root_chord_percent*RC
            D  = wing.Segments[i].root_chord_percent*RC
            E  = wing.Segments[i].percent_span_location/2 - wing.Segments[i-1].percent_span_location/2
            F  = wing.Segments[i-1].thickness_to_chord
            G  = wing.Segments[i].thickness_to_chord
            H  = wing.Segments[i-1].percent_span_location/2
            J  = wing.Segments[i].percent_span_location/2
            L  = (C-D)/E
            M  = (G-F)/E
            
            Y1 = wing.Segments[i-1].percent_span_location/2
            Y2 = wing.Segments[i].percent_span_location/2
            
            if C==D and F==G:

                WB1 = 1/(G*C)* (Y1/8 - Y1**3/6 )
                WB2 = 1/(G*C)* (Y2/8 - Y2**3/6 )
                              
            
            # I'm pretty sure the H^2 term is not correct. So only the top equation is right
            elif C!=D and F==G:

                WB1 = (1/(G)) * ((4*C**2 + 8*C*H*L + (4*H**2 - 1)*L**2)*np.log(C + H*L - L*Y1) +\
                                 2*L*Y1*(2*C + L*(2*H + Y1)))/(8*L**3)
                WB2 = (1/(G)) * ((4*C**2 + 8*C*H*L + (4*H**2 - 1)*L**2)*np.log(C + H*L - L*Y2) +\
                                 2*L*Y2*(2*C + L*(2*H + Y2)))/(8*L**3)
  
            elif C==D and F!=G:
                
                WB1 = (1/(C))*((4*F**2 + 8*F*H*M + 3*H**2 *M**2)*np.log(F + H*M - M*Y1) +\
                               2*M*Y1*(2*F + M*(2*H + Y1)))/(8*M**3)
                WB2 = (1/(C))*((4*F**2 + 8*F*H*M + 3*H**2 *M**2)*np.log(F + H*M - M*Y2) +\
                               2*M*Y2*(2*F + M*(2*H + Y2)))/(8*M**3)
            
            elif C!=D and F!=G:

                WB1 =  (M**2 *(4*C**2 + 8*C*H*L + 3*H**2 *L**2)*np.log(C + H*L - L*Y1) -\
                        L*(4*M*Y1*(F*L - C*M) + L*(4*F**2 + 8*F*H*M + 3*H**2 \
                                                   *M**2)*np.log(F + H*M - M*Y1)))/(8*L**2 *M**2 *(F*L - C*M))
                WB2 =  (M**2 *(4*C**2 + 8*C*H*L + 3*H**2 *L**2)*np.log(C + H*L - L*Y2) -\
                        L*(4*M*Y2*(F*L - C*M) + L*(4*F**2 + 8*F*H*M + 3*H**2 \
                                                   *M**2)*np.log(F + H*M - M*Y2)))/(8*L**2 *M**2 *(F*L - C*M))
                
            run_sum += (WB2-WB1)
            
        run_sum = run_sum*b**2 # This b^2 is because the Y is non dimensional
            
        weight_factor = gamma*run_sum
        
        weight = 4.22*(area/Units.feet**2) + (weight_factor/Units.lb)
            
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