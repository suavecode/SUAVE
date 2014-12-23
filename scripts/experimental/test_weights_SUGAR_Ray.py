# test_weights_SUGAR_Ray.py
# Tim Momose, November 2014
# NASA N+3 SUGAR Ray BWB Weight Breakdown Test
# In Progress



import SUAVE.Methods.Weights.Correlations as wt
from SUAVE.Attributes import Units
from math import exp, log
import pylab as plt

TOGW      = 182500.0 * Units.pounds
OEW_expct = 110493.0 * Units.pounds

b         = 2022.2 * Units.inches
b_ow      = b*(1-.22457)
bv        = 10.83 * Units.feet
l_cab     = 0.33025 * b
l_cab_out = 0.47245 * l_cab
c_center  = 0.43593 * b    # chord length at centerline
c_owr     = 0.268164 * b    # root chord of outer wing
c_owm     = 0.11757  * b    # chord at taper switch point
c_ref_r   = 264.462 * Units.inches
c_t       = 73.6 * Units.inches
b_ow1     = 0.048877 * b    # semispan up to taper switch point
b_ow2     = 0.31*b #b_ow - b_ow1    # semispan from taper switch to tip
S_g       = 4136. * Units.feet ** 2.0
S_g_ow    = b_ow1/2. * (c_owr+c_owm) + b_ow2/2. * (c_owm+c_t)
S_cb      = ((b-b_ow)/2.)*(c_center+c_owr)
S_cabin   = ((b-b_ow)/2.)*(l_cab+l_cab_out)
S_aft     = S_cb - S_cabin
Sh        = S_aft
Sv        = (90.8 * Units.feet ** 2.0) # area of each winglet / rudder
tcw       = 0.1
tcv       = 0.08
L_w       = 27.7  * Units.deg
L_v       = 39.5  * Units.deg
w_taper   = c_t/c_ref_r
aft_taper = 0.8
n_eng     = 2.0
T_sls     = 18800. * Units.force_pound
N_ult     = 3.75
num_crew  = 2.0
num_attd  = 4.0
pax       = 155.
wt_cargo  = 0.0
ctrl_type = "fully powered"
ac_type   = "medium-range"

cabin_wt  = wt.BWB.cabin(S_cabin,TOGW)
aft_wt    = wt.BWB.aft_centerbody(n_eng,S_aft,aft_taper,TOGW)
gear_wt   = wt.Tube_Wing.landing_gear(TOGW)
sys_wts   = wt.Tube_Wing.systems(pax,ctrl_type,Sh,Sv,S_g,ac_type)
sys_wt    = sys_wts.wt_systems
vert_wts  = wt.Tube_Wing.tail_vertical(Sv,N_ult,bv,TOGW,tcv,L_v,S_g,0,0.35)
vert_wt   = 2.0 * (vert_wts.wt_tail_vertical + vert_wts.wt_rudder)
pload_wts = wt.Tube_Wing.payload(TOGW,OEW_expct,pax,wt_cargo)
pload_wt  = pload_wts.payload
#wt_eng    = wt.Propulsion.engine_jet(T_sls)
#wt_eng    = n_eng * 1.6 * wt_eng # including factor for nacelle etc. 
wt_eng    = 15918. * Units.pound # given
wt_crew    = (190. + 50.) * num_crew * Units.pound  #Includes crew's luggage
wt_attd    = (170. + 40.) * num_attd * Units.pound  #Includes attendants' luggage

wt_zf     = OEW_expct + pload_wt + wt_crew + wt_attd
wing_wt   = wt.Tube_Wing.wing_main(S_g_ow,b_ow,w_taper,tcw,L_w,N_ult,TOGW*(S_g_ow/S_g),wt_zf*(S_g_ow/S_g))

OEW_calc  = cabin_wt + aft_wt + wing_wt + gear_wt + sys_wt + vert_wt + wt_eng + wt_crew + wt_attd
ZFW_calc  = OEW_calc + pload_wt

fig  = plt.figure("Weight Breakdown")
axes = fig.add_subplot(1,2,1)
weights = [aft_wt+cabin_wt+vert_wt,wing_wt,pload_wt+wt_crew+wt_attd,TOGW-ZFW_calc,1,\
           sys_wts.wt_opitems,sys_wts.wt_hyd_pnu+sys_wts.wt_ac+sys_wts.wt_apu,sys_wts.wt_furnish,\
           sys_wts.wt_avionics+sys_wts.wt_instruments,sys_wts.wt_elec,sys_wts.wt_flt_ctrl,wt_eng,gear_wt]
label = ['Centerbody and Tails','Outer Wing','Payload','Usable Fuel','Other',\
         'Operational Items','Pneumatics, AC, APU','Furnishings, Equipment',\
         'Avionics, Autopilot','Electrical','Flight Controls','Propulsion','Landing Gear']
plt.pie(weights, labels=label, autopct="%1.1f%%")
axes.axis('equal')
#plt.show()

print "== WEIGHT ESTIMATE FOR BWB-450 DESIGN CONCEPT =="
print "Expected OEW:            {0:.2f} lb ".format(OEW_expct / Units.pounds)
print "Calculated OEW:          {0:.2f} lb ({1:.1f}%)".format(OEW_calc / Units.pounds, 100*OEW_calc/OEW_expct)
print "  -Outer wing:           {0:.2f} lb ({1:.1f}%)".format(wing_wt/Units.pounds,100*wing_wt/Units.pounds/12500.)
print "  -Cabin:                {0:.2f} lb ({1:.1f}%)".format(cabin_wt/Units.pounds,100*(cabin_wt+aft_wt)/Units.pounds/41137.)
print "  -Aft-centerbody:       {0:.2f} lb".format(aft_wt/Units.pounds)
print "  -Vertical Surfaces:    {0:.2f} lb ({1:.1f}%)".format(vert_wt/Units.pounds,100*vert_wt/Units.pounds/904.)
print "  -Landing gear:         {0:.2f} lb ({1:.1f}%)".format(gear_wt/Units.pounds,100*gear_wt/Units.pounds/7198.)
print "  -GNC systems, opitems: {0:.2f} lb ({1:.1f}%)".format(sys_wt/Units.pounds,100*sys_wt/Units.pounds/(186.+3553.+9080.+3225.+1079.+3346.+6015.))
print "  -Engines:              {0:.2f} lb ({1:.1f}%)".format(wt_eng/Units.pounds,100*wt_eng/Units.pounds/15918.)
print "  -Attendants and Crew:  {0:.2f} lb ".format((wt_crew+wt_attd)/Units.pounds)
print "Calculated ZFW:          {0:.2f} lb".format((ZFW_calc)/Units.pounds)
print "Available fuel weight:   {0:.2f} lb".format((TOGW - OEW_calc - pload_wt)/Units.pounds)

print " "