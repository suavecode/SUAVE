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
def wing_main(wing,Nult,TOW,wt_zf):
    """ Calculate the wing weight of the aircraft based on the fully-stressed 
    bending weight of the wing box
    
    Assumptions:
        calculated total wing weight based on a bending index and actual data 
        from 15 transport aircraft 
    
    Source: 
        N/A
        
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
    span  = wing.spans.projected  / Units.ft # Convert meters to ft
    taper = wing.taper
    sweep = wing.sweeps.quarter_chord
    area  = wing.areas.reference / Units.ft**2 # Convert meters squared to ft squared
    t_c_w = wing.thickness_to_chord
    mtow  = TOW / Units.lb # Convert kg to lbs
    zfw   = wt_zf / Units.lb # Convert kg to lbs
    
    
    if len(wing.Segments)>0:
        RC = wing.chords.root
        run_sum = 0
        b = span
        for i in range(1,len(wing.Segments)):
            C  = wing.Segments[i-1].root_chord_percent*RC
            D  = wing.Segments[i].root_chord_percent*RC
            E  = wing.Segments[i].percent_span_location - wing.Segments[i-1].percent_span_location
            F  = wing.Segments[i-1].thickness_to_chord
            G  = wing.Segments[i].thickness_to_chord
            H  = wing.Segments[i-1].percent_span_location
            
            Y1 = wing.Segments[i].percent_span_location
            Y2 = wing.Segments[i-1].percent_span_location
            
            
            WB1 = -(E*(-4*E*G**2 *Y1*C**2 + 4*E*F*G*Y1*C**2 + 4*E**2 *F**2 *np.log(-E*F - (F - G)*(H - Y1))*C**2 -\
                       F**2 *np.log(-E*F - (F - G)*(H - Y1))*C**2 - G**2 *np.log(-E*F - (F - G)*(H - Y1))*C**2 +\
                       4*F**2 *H**2 *np.log(-E*F - (F - G)*(H - Y1))*C**2 + 4*G**2 *H**2 *\
                       np.log(-E*F - (F - G)*(H - Y1))*C**2 - 8*F*G*H**2 *np.log(-E*F - (F - G)*(H - Y1))*C**2 + \
                       2 *F*G *np.log(-E*F - (F - G)*(H - Y1))*C**2 + 8*E*F**2 *H *np.log(-E*F - (F - G)*(H - Y1))*C**2\
                       - 8*E*F*G*H*np.log(-E*F - (F - G)*(H - Y1))*C**2 - 4*E**2 *F**2 *np.log(D*(H - Y1) - \
                       C*(E + H - Y1))*C**2 + F**2 *np.log(D*(H - Y1) - C*(E + H - Y1))*C**2 - 4*E**2 *G**2 *np.log(D*(H - Y1)\
                       - C*(E + H - Y1))*C**2 + G**2 *np.log(D*(H - Y1) - C*(E + H - Y1))*C**2 - 4*F**2 *H**2 *np.log(D*(H - Y1) \
                       - C*(E + H - Y1))*C**2 - 4*G**2 *H**2 *np.log(D*(H - Y1) - C*(E + H - Y1))*C**2 + 8*F*G*H**2 *np.log(D*(H - Y1) \
                       - C*(E + H - Y1))*C**2 + 8*E**2 *F*G*np.log(D*(H - Y1) - C*(E + H - Y1))*C**2 - 2 *F*G*np.log(D*(H - Y1) \
                       - C*(E + H - Y1))*C**2 - 8*E*F**2 *H*np.log(D*(H - Y1) - C*(E + H - Y1))*C**2 - 8 *E*G**2 *H*np.log(D*(H - Y1)\
                       - C*(E + H - Y1))*C**2 + 16*E*F*G*H*np.log(D*(H - Y1) - C*(E + H - Y1))*C**2 - 4*D*E*F**2 *Y1*C + 4*D*E*G**2 *Y1*C \
                       - 8*D*E**2 *F**2 *np.log(-E*F - (F - G)*(H - Y1))*C + 2 *D*F**2 *np.log(-E*F - (F - G)*(H - Y1))*C +\
                       2*D*G**2 *np.log(-E*F - (F - G)*(H - Y1))*C - 8*D*F**2 *H**2 *np.log(-E*F - (F - G)*(H - Y1))*C - 8*D*G**2 *H**2 \
                       *np.log(-E*F - (F - G)*(H - Y1))*C + 16*D*F*G*H**2 *np.log(-E*F - (F - G)*(H - Y1))*C -\
                       4*D*F*G*np.log(-E*F - (F - G)*(H - Y1))*C - 16*D*E*F**2 *H*np.log(-E*F - (F - G)*(H - Y1))*C + \
                       16*D*E*F*G*H*np.log(-E*F - (F - G)*(H - Y1))*C - 2 *D*F**2 *np.log(D*(H - Y1) - C*(E + H - Y1))*C -\
                       2 *D*G**2 *np.log(D*(H - Y1) - C*(E + H - Y1))*C + 8*D*F**2 *H**2 *np.log(D*(H - Y1) - C*(E + H - Y1))*C + \
                       8*D*G**2 *H**2 *np.log(D*(H - Y1) - C*(E + H - Y1))*C - 16*D*F*G*H**2 *np.log(D*(H - Y1) - C*(E + H - Y1))*C + \
                       4*D*F*G*np.log(D*(H - Y1) - C*(E + H - Y1))*C + 8*D*E*F**2 *H*np.log(D*(H - Y1) - C*(E + H - Y1))*C +\
                       8*D*E*G**2 *H*np.log(D*(H - Y1) - C*(E + H - Y1))*C - 16*D*E*F*G*H*np.log(D*(H - Y1) - C*(E + H - Y1))*C + \
                       4*D**2 *E*F**2 *Y1 - 4*D**2 *E*F*G*Y1 - D**2 *F**2 *np.log(-E*F - (F - G)*(H - Y1)) + 4*D**2 *E**2 *F**2 \
                       *np.log(-E*F - (F - G)*(H - Y1)) - D**2 *G**2 *np.log(-E*F - (F - G)*(H - Y1)) + 4*D**2 *F**2 *H**2\
                       *np.log(-E*F - (F - G)*(H - Y1)) + 4*D**2 *G**2 *H**2 *np.log(-E*F - (F - G)*(H - Y1)) - 8*D**2 *F*G*H**2 \
                       *np.log(-E*F - (F - G)*(H - Y1)) + 2 *D**2 *F*G *np.log(-E*F - (F - G)*(H - Y1)) + 8*D**2 *E*F**2 *H \
                       *np.log(-E*F - (F - G)*(H - Y1)) - 8*D**2 *E*F*G*H*np.log(-E*F - (F - G)*(H - Y1)) + D**2 *F**2 \
                       *np.log(D*(H - Y1) - C*(E + H - Y1)) + D**2 *G**2 *np.log(D*(H - Y1) - C*(E + H - Y1)) - 4*D**2 *F**2 *H**2\
                       *np.log(D*(H - Y1) - C*(E + H - Y1)) - 4*D**2 *G**2 *H**2 *np.log(D*(H - Y1) - C*(E + H - Y1)) + 8*D**2 *F*G*H**2\
                       *np.log(D*(H - Y1) - C*(E + H - Y1)) - 2 *D**2 *F*G *np.log(D*(H - Y1) - C*(E + H - Y1))))/(8*(C - D)**2 *(F - G)**2 *(C*G - D*F))
            
            
            WB2 = -(E*(-4*E*G**2 *Y2*C**2 + 4*E*F*G*Y2*C**2 + 4*E**2 *F**2 *np.log(-E*F - (F - G)*(H - Y2))*C**2 -\
                       F**2 *np.log(-E*F - (F - G)*(H - Y2))*C**2 - G**2 *np.log(-E*F - (F - G)*(H - Y2))*C**2 +\
                       4*F**2 *H**2 *np.log(-E*F - (F - G)*(H - Y2))*C**2 + 4*G**2 *H**2 *\
                       np.log(-E*F - (F - G)*(H - Y2))*C**2 - 8*F*G*H**2 *np.log(-E*F - (F - G)*(H - Y2))*C**2 + \
                       2 *F*G *np.log(-E*F - (F - G)*(H - Y2))*C**2 + 8*E*F**2 *H *np.log(-E*F - (F - G)*(H - Y2))*C**2\
                       - 8*E*F*G*H*np.log(-E*F - (F - G)*(H - Y2))*C**2 - 4*E**2 *F**2 *np.log(D*(H - Y2) - \
                       C*(E + H - Y2))*C**2 + F**2 *np.log(D*(H - Y2) - C*(E + H - Y2))*C**2 - 4*E**2 *G**2 *np.log(D*(H - Y2)\
                       - C*(E + H - Y2))*C**2 + G**2 *np.log(D*(H - Y2) - C*(E + H - Y2))*C**2 - 4*F**2 *H**2 *np.log(D*(H - Y2) \
                       - C*(E + H - Y2))*C**2 - 4*G**2 *H**2 *np.log(D*(H - Y2) - C*(E + H - Y2))*C**2 + 8*F*G*H**2 *np.log(D*(H - Y2) \
                       - C*(E + H - Y2))*C**2 + 8*E**2 *F*G*np.log(D*(H - Y2) - C*(E + H - Y2))*C**2 - 2 *F*G*np.log(D*(H - Y2) \
                       - C*(E + H - Y2))*C**2 - 8*E*F**2 *H*np.log(D*(H - Y2) - C*(E + H - Y2))*C**2 - 8 *E*G**2 *H*np.log(D*(H - Y2)\
                       - C*(E + H - Y2))*C**2 + 16*E*F*G*H*np.log(D*(H - Y2) - C*(E + H - Y2))*C**2 - 4*D*E*F**2 *Y2*C + 4*D*E*G**2 *Y2*C \
                       - 8*D*E**2 *F**2 *np.log(-E*F - (F - G)*(H - Y2))*C + 2 *D*F**2 *np.log(-E*F - (F - G)*(H - Y2))*C +\
                       2*D*G**2 *np.log(-E*F - (F - G)*(H - Y2))*C - 8*D*F**2 *H**2 *np.log(-E*F - (F - G)*(H - Y2))*C - 8*D*G**2 *H**2 \
                       *np.log(-E*F - (F - G)*(H - Y2))*C + 16*D*F*G*H**2 *np.log(-E*F - (F - G)*(H - Y2))*C -\
                       4*D*F*G*np.log(-E*F - (F - G)*(H - Y2))*C - 16*D*E*F**2 *H*np.log(-E*F - (F - G)*(H - Y2))*C + \
                       16*D*E*F*G*H*np.log(-E*F - (F - G)*(H - Y2))*C - 2 *D*F**2 *np.log(D*(H - Y2) - C*(E + H - Y2))*C -\
                       2 *D*G**2 *np.log(D*(H - Y2) - C*(E + H - Y2))*C + 8*D*F**2 *H**2 *np.log(D*(H - Y2) - C*(E + H - Y2))*C + \
                       8*D*G**2 *H**2 *np.log(D*(H - Y2) - C*(E + H - Y2))*C - 16*D*F*G*H**2 *np.log(D*(H - Y2) - C*(E + H - Y2))*C + \
                       4*D*F*G*np.log(D*(H - Y2) - C*(E + H - Y2))*C + 8*D*E*F**2 *H*np.log(D*(H - Y2) - C*(E + H - Y2))*C +\
                       8*D*E*G**2 *H*np.log(D*(H - Y2) - C*(E + H - Y2))*C - 16*D*E*F*G*H*np.log(D*(H - Y2) - C*(E + H - Y2))*C + \
                       4*D**2 *E*F**2 *Y2 - 4*D**2 *E*F*G*Y2 - D**2 *F**2 *np.log(-E*F - (F - G)*(H - Y2)) + 4*D**2 *E**2 *F**2 \
                       *np.log(-E*F - (F - G)*(H - Y2)) - D**2 *G**2 *np.log(-E*F - (F - G)*(H - Y2)) + 4*D**2 *F**2 *H**2\
                       *np.log(-E*F - (F - G)*(H - Y2)) + 4*D**2 *G**2 *H**2 *np.log(-E*F - (F - G)*(H - Y2)) - 8*D**2 *F*G*H**2 \
                       *np.log(-E*F - (F - G)*(H - Y2)) + 2 *D**2 *F*G *np.log(-E*F - (F - G)*(H - Y2)) + 8*D**2 *E*F**2 *H \
                       *np.log(-E*F - (F - G)*(H - Y2)) - 8*D**2 *E*F*G*H*np.log(-E*F - (F - G)*(H - Y2)) + D**2 *F**2 \
                       *np.log(D*(H - Y2) - C*(E + H - Y2)) + D**2 *G**2 *np.log(D*(H - Y2) - C*(E + H - Y2)) - 4*D**2 *F**2 *H**2\
                       *np.log(D*(H - Y2) - C*(E + H - Y2)) - 4*D**2 *G**2 *H**2 *np.log(D*(H - Y2) - C*(E + H - Y2)) + 8*D**2 *F*G*H**2\
                       *np.log(D*(H - Y2) - C*(E + H - Y2)) - 2 *D**2 *F*G *np.log(D*(H - Y2) - C*(E + H - Y2))))/(8*(C - D)**2 *(F - G)**2 *(C*G - D*F))
           

            
     
            
            run_sum += (WB2-WB1)
            
        weight = 4.22*area 
            
    else:

        #Calculate weight of wing for traditional aircraft wing
        weight = 4.22*area + 1.642*10.**-6. * Nult*(span)**3. *(mtow*zfw)**0.5 \
                 * (1.+2.*taper)/(t_c_w*(np.cos(sweep))**2. * area*(1.+taper) )
             
    
    weight = weight * Units.lb # Convert lb to kg

    return weight