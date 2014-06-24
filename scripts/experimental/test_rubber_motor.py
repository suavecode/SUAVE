#Created 4/8/14 Tim MacDonald
#Last Modified 6/19/14 Tim MacDonald
#AIAA 2014-0536
#Electric Propulsion Modeling for Conceptual Aircraft Design
#Robert A. McDonald

import SUAVE
from SUAVE.Components.Energy.Converters import MotorMap
import matplotlib.pyplot as pl
import numpy as np

def main():
	
	# Test Case or Rubber Motor
	rflag = 0
	
	
	# Specify motor parameters
	n = 100 # used for linspace
	if rflag == 0: # Initial test case in paper
		omegah = 4500 #rpm
		Qh = 125 # Nm (Torque)
		etah = .941 # Percent/100
		k0 = 0.95
		kQ = 2
		kP = 2
		komega = 2
	elif rflag == 1: # Normalized map
		omegah = 1
		Qh = 1
		etah = 0.95
		k0 = .5
		kQ = 2
		kP = 2
		komega = 2
	elif rflag == 2: # Rubber Motor (a general case - not empirical)
		Q_rated = 12.2e4
		omega_limit = 8500
		k0 = .5
		kQ = 2
		kP = 2
		komega = 2
		Qh = Q_rated/kQ
		omegah = omega_limit/komega
		P_rated = kP*omegah*Qh
		etah = .95
		
	
	
	# Calculate bounds on motor maps
	Q_rated = kQ*Qh
	P_rated = kP*omegah*Qh
	omega_limit = komega*omegah
	omega_rated = kP/kQ*omegah # This is the point where power becomes limiting
	
	# Power requirements based Q_rated and E3 engine concept
	# Line through common sizing points
	omega_E3 = np.linspace(omega_rated*.75,omega_rated,n)
	Q_E3 = np.linspace(0.123*Q_rated,Q_rated,n)
	
	# Determine motor map parameters
	C0 = k0*omegah*Qh/6*(1-etah)/etah
	RubberMotor = MotorMap.GenConverter()
	RubberMotor(omegah,Qh,etah,k0,kQ,kP,komega)
	print "omegah = %f, Qh = %f, etah = %f" %(RubberMotor.omegah,RubberMotor.Qh,RubberMotor.etah)
	print "C0 = %f, C1 = %f, C2 = %.8f, C3 = %f" %(RubberMotor.C0,RubberMotor.C1,RubberMotor.C2,RubberMotor.C3)
	
	# Check single efficiency value
	omega = 500
	Q = 25
	etap = RubberMotor.eta(omega,Q)
	print "etap = %.3f" %(etap)
	
	# Create full motor map
	#
	
	# Create rated bounds on the map
	omega = np.linspace(0,omega_limit*1.1,n)
	q = np.linspace(0,Q_rated*1.1,n)
	omega_prate = np.linspace(omega_rated,omega_limit,n)
	q_prate = P_rated/omega_prate
	Om,Q = np.meshgrid(omega,q)
	pl.figure(1)
	#pl.axes = ([0,omega_limit*1.1,0,Q_rated*1.1])
	#pl.contourf(Om,Q,RubberMotor.eta(Om,Q),levels=[.65,.7,.75,.8,.85,.875,.9,.925,.94,1],cmap=pl.cm.RdYlGn)
	#pl.contourf(Om,Q,RubberMotor.eta(Om,Q),levels=[0,.1,.2,.3,.4,.5,.6,.65],cmap=pl.cm.OrRd)
	if rflag == 0: # Special values used to match test case in paper
		C = pl.contour(Om,Q,RubberMotor.eta(Om,Q),colors='black',linewidth=.5,levels=[.65,.7,.75,.8,.85,.875,.9,.925,.94])
	else:
		C = pl.contour(Om,Q,RubberMotor.eta(Om,Q),colors='black',linewidth=.5,levels=[.25,.5,.75,.8,.85,.9,.93,.94,.949])
	p1,=pl.plot([omega_limit,omega_limit],[0,P_rated/omega_limit],label="line 1")	
	p2,=pl.plot([0,omega_rated],[Q_rated,Q_rated])
	p3,=pl.plot(omega_prate,q_prate)
	pl.legend([p2,p3,p1],["Rated Torque","Rated Power","Rated RPM"])
	#if rflag == 2:
	p4,=pl.plot(omega_E3,Q_E3)
	#om_lst = np.array([3100,3600,3750,3700,omega_rated])*omega_rated/4250.0
	#Q_lst = np.array([2.2e4,5.1e4,7e4,7.3e4,Q_rated])*Q_rated/12.2e4
	# List of common sizing parameters
	om_lst = np.array([0.729,0.847,0.882,0.871,1])*omega_rated
	Q_lst = np.array([0.18,0.418,0.57,0.60,1])*Q_rated	
	pl.scatter(om_lst, Q_lst, s=20, c='c', marker='o')
	pl.legend([p2,p3,p1,p4],["Rated Torque","Rated Power","Rated RPM","Common Sizing Values"])
	pl.clabel(C,inline=1,fontsize=10)
	#pl.xticks(())
	#pl.yticks(())
	pl.xlabel("RPM")
	pl.ylabel("Torque (Nm)")
	pl.title("Efficiency")
	#pl.show()
	
	# Efficiency plot along common sizing line
	pl.figure(2)
	p_line = omega_E3*Q_E3/60 # divide by 60 for RPS
	p3,=pl.plot(p_line,RubberMotor.eta(omega_E3,Q_E3))
	#pl.scatter(om_lst*Q_lst/60,RubberMotor.eta(om_lst, Q_lst),s=20, c='c', marker='o')
	pl.xlabel("Power (J/s)")
	pl.ylabel("Efficiency")	
	pl.show()
	
	

if __name__ == '__main__':
    main()