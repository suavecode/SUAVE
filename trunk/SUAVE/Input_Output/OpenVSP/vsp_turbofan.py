## @ingroup Input_Output-OpenVSP
# vsp_turbofan.py
# 
# Created:  Jul 2016, T. MacDonald
# Modified: Jun 2017, T. MacDonald
#           Jul 2017, T. MacDonald
#           Oct 2018, T. MacDonald
#           Nov 2018, T. MacDonald
#           Jan 2019, T. MacDonald
#           Jan 2020, T. MacDonald 
#           Mar 2020, M. Clarke
#           May 2020, E. Botero
#           Jul 2020, E. Botero 
#           Feb 2021, T. MacDonald
#           May 2021, E. Botero 

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

## @ingroup Input_Output-OpenVSP  
def write_vsp_turbofan(turbofan, OML_set_ind):
    """This converts turbofans into OpenVSP format.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    turbofan.
      number_of_engines                       [-]
      engine_length                           [m]
      nacelle_diameter                        [m]
      origin                                  [m] in all three dimension, should have as many origins as engines
      OpenVSP_flow_through                    <boolean> if True create a flow through nacelle, if False create a cylinder

    Outputs:
    Operates on the active OpenVSP model, no direct output

    Properties Used:
    N/A
    """    
    n_engines   = turbofan.number_of_engines
    length      = turbofan.engine_length
    width       = turbofan.nacelle_diameter
    origins     = turbofan.origin
    inlet_width = turbofan.inlet_diameter
    tf_tag      = turbofan.tag

    # True will create a flow-through subsonic nacelle (which may have dimensional errors)
    # False will create a cylindrical stack (essentially a cylinder)
    ft_flag = turbofan.OpenVSP_flow_through

    import operator # import here since engines are not always needed
    # sort engines per left to right convention
    origins_sorted = sorted(origins, key=operator.itemgetter(1))

    for ii in range(0,int(n_engines)):

	origin = origins_sorted[ii]

	x = origin[0]
	y = origin[1]
	z = origin[2]

	if ft_flag == True:
	    nac_id = vsp.AddGeom( "BODYOFREVOLUTION")
	    vsp.SetGeomName(nac_id, tf_tag+'_'+str(ii+1))

	    # Origin
	    vsp.SetParmVal(nac_id,'X_Location','XForm',x)
	    vsp.SetParmVal(nac_id,'Y_Location','XForm',y)
	    vsp.SetParmVal(nac_id,'Z_Location','XForm',z)
	    vsp.SetParmVal(nac_id,'Abs_Or_Relitive_flag','XForm',vsp.ABS) # misspelling from OpenVSP  

	    # Length and overall diameter
	    vsp.SetParmVal(nac_id,"Diameter","Design",inlet_width)

	    vsp.ChangeBORXSecShape(nac_id ,vsp.XS_SUPER_ELLIPSE)
	    vsp.Update()
	    vsp.SetParmVal(nac_id, "Super_Height", "XSecCurve", (width-inlet_width)/2)
	    vsp.SetParmVal(nac_id, "Super_Width", "XSecCurve", length)
	    vsp.SetParmVal(nac_id, "Super_MaxWidthLoc", "XSecCurve", -1.)
	    vsp.SetParmVal(nac_id, "Super_M", "XSecCurve", 2.)
	    vsp.SetParmVal(nac_id, "Super_N", "XSecCurve", 1.)             

	else:
	    nac_id = vsp.AddGeom("STACK")
	    vsp.SetGeomName(nac_id, tf_tag+'_'+str(ii+1))

	    # Origin
	    vsp.SetParmVal(nac_id,'X_Location','XForm',x)
	    vsp.SetParmVal(nac_id,'Y_Location','XForm',y)
	    vsp.SetParmVal(nac_id,'Z_Location','XForm',z)
	    vsp.SetParmVal(nac_id,'Abs_Or_Relitive_flag','XForm',vsp.ABS) # misspelling from OpenVSP
	    vsp.SetParmVal(nac_id,'Origin','XForm',0.5)            

	    vsp.CutXSec(nac_id,2) # remove extra default subsurface
	    xsecsurf = vsp.GetXSecSurf(nac_id,0)
	    vsp.ChangeXSecShape(xsecsurf,1,vsp.XS_CIRCLE)
	    vsp.ChangeXSecShape(xsecsurf,2,vsp.XS_CIRCLE)
	    vsp.Update()
	    vsp.SetParmVal(nac_id, "Circle_Diameter", "XSecCurve_1", width)
	    vsp.SetParmVal(nac_id, "Circle_Diameter", "XSecCurve_2", width)
	    vsp.SetParmVal(nac_id, "XDelta", "XSec_1", 0)
	    vsp.SetParmVal(nac_id, "XDelta", "XSec_2", length)
	    vsp.SetParmVal(nac_id, "XDelta", "XSec_3", 0)

	vsp.SetSetFlag(nac_id, OML_set_ind, True)

	vsp.Update()
