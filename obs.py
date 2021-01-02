
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ADM-based tools
# Melissa Munoz
# Updated Dec 2020
#
# See publication Munoz et al. 2020 
# See also Owocki et al. 2006 for more details on the ADM formalism
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#-------------------------------------------------------------------------------
# Library import ---------------------------------------------------------------
#-------------------------------------------------------------------------------

import numpy as np
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.integrate import trapz
from scipy.integrate import cumtrapz
from scipy.special import voigt_profile
import copy

#-------------------------------------------------------------------------------
# Some constants ---------------------------------------------------------------
#-------------------------------------------------------------------------------

G = 6.67408*10**(-8)
eV = 1.602*10**(-12)
c = 2.99792458*10**(10)	 
h= 6.6260755*10**(-27)	 
kb = 1.380658*10**(-16)
eV = 1.6021772*10**(-12)	
me = 9.1093897*10**(-28)
mp = 1.6726*10**(-24)
mH = 1.6733*10**(-24)
e = 4.8032068*10**(-10)
X=0.72
Y=0.28
alphae=(1.+X)/2.#0.5
alphap=(X+Y/4.)#0.72#1./0.72
sigmat = 6.6524*10**(-25)
sigma0=sigmat*3./(16.*np.pi)
Msol = 1.99*10**(33)
Rsol = 6.96*10**(10)
Lsol = 3.9*10**(33)


#-------------------------------------------------------------------------------
# Line-of-sight angle ----------------------------------------------------------
#-------------------------------------------------------------------------------

#With degenerancy between inclination and obliquity
def csalpha(phi, beta, inc):
	return np.sin(beta)*np.cos(phi)*np.sin(inc)+np.cos(beta)*np.cos(inc)

#To avoid the degenrancy, the Independant paramters A and B can be used, where
#A = inc + beta,
#B = |inc - beta|
def csalpha2(phi, A, B):
	return 0.5*(np.cos(B)*(1.+np.cos(phi)) + np.cos(A)*(1.-np.cos(phi)) )



#-------------------------------------------------------------------------------
#Parametric equations for the Stokes Q and U parameters (see Fox 1992) ---------
#-------------------------------------------------------------------------------

#Axi-symmetric envelope
def QUp_(phi, amp, inc, beta, thetaIS, QIS, UIS):
	inc = np.radians(inc)
	beta = np.radians(beta)
	lda = phi*2.*np.pi

	q0= 0.5*amp*np.sin(inc)**2*(3.*np.cos(beta)**2-1.)*np.cos(thetaIS) + QIS #QIS*np.cos(thetaIS) - UIS*np.sin(thetaIS) +
	q1= amp*np.sin(inc)*np.sin(2.*beta)*np.sin(thetaIS) 
	q2= 0.5*amp*np.sin(2.*inc)*np.sin(2.*beta)*np.cos(thetaIS)
	q3=-0.5*amp*(1.+np.cos(inc)**2)*np.sin(beta)**2*np.cos(thetaIS)
	q4= amp*np.cos(inc)*np.sin(beta)**2*np.sin(thetaIS)
	Q = q0 + q1*np.cos(lda) + q2*np.sin(lda) + q3*np.cos(2.*lda) +  q4*np.sin(2.*lda)

	u0= 0.5*amp*np.sin(inc)**2*(3.*np.cos(beta)**2-1)*np.sin(thetaIS) + UIS #QIS*np.sin(thetaIS) + UIS*np.cos(thetaIS) 
	u1=-amp*np.sin(inc)*np.sin(2.*beta)*np.cos(thetaIS) 
	u2= 0.5*amp*np.sin(2.*inc)*np.sin(2.*beta)*np.sin(thetaIS)
	u3=-0.5*amp*(1.+np.cos(inc)**2)*np.sin(beta)**2*np.sin(thetaIS)
	u4=-amp*np.cos(inc)*np.sin(beta)**2*np.cos(thetaIS)
	U = u0 + u1*np.cos(lda) + u2*np.sin(lda) + u3*np.cos(2.*lda) +  u4*np.sin(2.*lda)
	return [Q,U]

#General envelope
def QU_(ph,amp,taugamma1,taugamma2,taugamma3,taugamma4,inc,beta,thetaIS,QIS,UIS):
	inc = np.radians(inc)
	beta = np.radians(beta)
	lda = 2.*np.pi*ph

	V = amp
	W = taugamma1
	X = taugamma2
	Y = taugamma3
	Z = taugamma4

	E = np.sin(2.*inc)*np.cos(thetaIS)
	F = 2.*np.sin(inc)*np.sin(thetaIS)
	J = np.sin(2.*inc)*np.sin(thetaIS)
	K = 2.*np.sin(inc)*np.cos(thetaIS)
	L = (1.+np.cos(inc)**2)*np.cos(thetaIS)
	M = 2.*np.cos(inc)*np.sin(thetaIS)
	N = (1.+np.cos(inc)**2)*np.sin(thetaIS)
	R = 2.*np.cos(inc)*np.cos(thetaIS)

	q0 = V*np.sin(inc)**2*(3.*np.cos(beta)**2 - 1.)*np.cos(thetaIS) + 3.*X*np.sin(inc)**2*np.sin(2.*beta)*np.cos(thetaIS) + 3.*Y*np.sin(inc)**2*np.sin(2.*beta)*np.cos(thetaIS) + QIS
	q1 = 2.*W*E*np.cos(beta) - 2.*Z*E*np.sin(beta) + V*F*np.sin(2.*beta) - 2.*X*F*np.cos(2.*beta) - Y*F*np.sin(2.*beta)
	q2 = V*E*np.sin(2.*beta) - 2.*X*E*np.cos(2.*beta) - Y*E*np.sin(2.*beta) - 2.*W*F*np.cos(beta) - 2.*Z*F*np.sin(beta)
	q3 = -V*L*np.sin(beta)**2 + X*L*np.sin(2.*beta) - Y*L*(1.+np.cos(beta)**2) + 2.*W*M*np.sin(beta) - 2.*Z*M*np.cos(beta)
	q4 = 2.*W*L*np.sin(beta) + 2.*Z*L*np.cos(beta) + V*M*np.sin(beta)**2 - X*M*np.sin(2.*beta) + Y*M*(1+np.cos(beta)**2)
	Q = q0 + q1*np.cos(lda) + q2*np.sin(lda) + q3*np.cos(2.*lda) + q4*np.sin(2.*lda) 

	u0 = V*np.sin(inc)**2*(3.*np.cos(beta)**2 - 1.)*np.sin(thetaIS) + 3.*X*np.sin(inc)**2*np.sin(2.*beta)*np.sin(thetaIS) + 3.*Y*np.sin(inc)**2*np.sin(2.*beta)*np.sin(thetaIS) + UIS
	u1 = 2.*W*J*np.cos(beta) - 2.*Z*J*np.sin(beta) - V*K*np.sin(2.*beta) + 2.*X*K*np.cos(2.*beta) + Y*K*np.sin(2.*beta)
	u2 = V*J*np.sin(2.*beta) - 2.*X*J*np.cos(2.*beta) - Y*J*np.sin(2.*beta) + 2.*W*K*np.cos(beta) + 2.*Z*K*np.sin(beta)
	u3 = -V*N*np.sin(beta)**2 + X*N*np.sin(2.*beta) - Y*N*(1.+np.cos(beta)**2) - 2.*W*R*np.sin(beta) + 2.*Z*R*np.cos(beta)
	u4 = 2.*W*N*np.sin(beta) + 2.*Z*N*np.cos(beta) - V*R*np.sin(beta)**2 + X*R*np.sin(2.*beta) - Y*R*(1+np.cos(beta)**2)
	U = u0 + u1*np.cos(lda) + u2*np.sin(lda) + u3*np.cos(2.*lda) + u4*np.sin(2.*lda) 

	return [Q,U]

#-------------------------------------------------------------------------------
#Equivalendt witch calculation from a line profile -----------------------------
#-------------------------------------------------------------------------------

def ew(x,y,a,b,yerr='None'):

	if yerr != 'None':
		integrand=1.-y
		integranderr=yerr

		index=np.arange(len(x))
		index=index[ (x>a) & (x<b)]

		EW=0.
		errEW=0.
		for k in (index-1):
			EW+=0.5*(x[k+1]-x[k])*(integrand[k+1]+integrand[k])
			errEW+=0.5*(x[k+1]-x[k])*(integranderr[k+1]+integranderr[k])
		
		out=[-EW,errEW]

	else:
		integrand=1.-y

		index=np.arange(len(x))
		index=index[ (x>a) & (x<b) ]

		EW=0.
		for k in (index-1):
			#print( k, x[k+1],x[k], integrand[k+1],integrand[k])
			EW+=0.5*(x[k+1]-x[k])*(integrand[k+1]+integrand[k])
			
		out=-EW

	return out

#-------------------------------------------------------------------------------
# Energy levels ----------------------------------------------------------------
#-------------------------------------------------------------------------------

def gn(n):
	return 2.*n**2

def En(n): 
	return -13.6*eV/n**2

g2=gn(2)
f32=0.64108
const1=np.pi*e**2*h**3*f32*g2/(me*c*2*(2.*np.pi*me*kb)**(3./2.))


#-------------------------------------------------------------------------------
# Line profiles ----------------------------------------------------------------
#-------------------------------------------------------------------------------

def phi_L(x,nuN):
	#Lorentzian line profile
	#x=nu-nu0
	return (nuN)/(x**2 + (nuN)**2)/np.pi

def phi_G(x,nuD):
	#Gaussian line profile
	#x=nu-nu0
	return np.exp(-(x/nuD)**2)/(nuD*np.sqrt(np.pi))

def phi_LG(x,nuD,nuN):
	#Voigt line profile
	#x=nu-nu0
	return voigt_profile(x, nuD/np.sqrt(2.), nuN)



#-------------------------------------------------------------------------------
# Halpha line opacity ----------------------------------------------------------
#-------------------------------------------------------------------------------

def chi32(x,nuD,nuN,Np,Ne,T):
	b2=1. 
	b3=1.
	return const1*Np*Ne*T**(-1.5)*(b2*np.exp(39455.1/T) - b3*np.exp(17535.6/T))*phi_LG(x,nuD,nuN)
	 	 
def Bnu(nu,T):
	return 2.*h*nu**3/c**2/(np.exp(h*nu/(kb*T))-1.)



#-------------------------------------------------------------------------------
# RT ---------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# Light curve synthesis (optically thin approximation)
#-------------------------------------------------------------------------------
def LC(phi, A, B, rhow, rhoc, rhoh, Rstar, Rc, dm0 ):
	
	#t0 = time.time()

	#Defining phase grid 
	phi=np.concatenate(([0.],phi))
	PH=len(phi)

	N = np.shape(rhow)
	NNx = N[1]
	NNy = N[2]
	NNz = N[0]

	#Defining spatial grids
	XX=np.linspace(-Rc,Rc,NNx)
	YY=np.linspace(-Rc,Rc,NNy)
	ZZ=np.linspace(-Rc,Rc,NNz)
	dX=np.abs(XX[0]-XX[1])
	dY=np.abs(YY[0]-YY[1])
	dZ=np.abs(ZZ[0]-ZZ[1])
	dx=dX*Rstar*Rsol
	dy=dY*Rstar*Rsol
	dz=dZ*Rstar*Rsol

	#Variable setup
	I0=np.ones([NNx,NNx]) #Intensity
	F1=np.zeros(PH) #Flux
	F2=np.zeros(PH) #Core fux
	dm=np.zeros(PH) #Differential magnitude
	for ph in range(PH):
		
		#Rotation of density cube 
		RHO=rhoh+rhoc+rhow
		RHO_rot = np.zeros([NNz,NNx,NNy])
		alpha=np.arccos(csalpha2(phi[ph]*2.*np.pi,np.radians(A),np.radians(B)))
		for k in range(0,NNx):
			RHO_rot[:,k,:]  = rotate(RHO[:,k,:],np.degrees(alpha),reshape=False)

		#Defining 3D meshgrid 
		Z_grid, X_grid, Y_grid = np.meshgrid( ZZ, XX, YY, indexing='xy')
		R_grid = np.sqrt( Z_grid**2 + X_grid**2 + Y_grid**2 )
		RHO_rot[ R_grid<1.0 ] = 0.
		RHO_rot[ (np.sqrt(Z_grid**2+Y_grid**2)<1) & (X_grid<0) ] = 0 #removing occulted regions 

		#Defining 2D meshgrid
		X_grid, Y_grid = np.meshgrid( XX, YY, indexing='xy')
		P_grid = np.sqrt( X_grid**2 + Y_grid**2 )
		I0[ P_grid > 1.0] = 0. #including only core rays

		#Computing electron density, optical depth and attenuated intensity
		ne = RHO_rot*alphae/mp
		dtau = dz*ne*sigmat
		tauinf = simps(dtau,axis=0) 
		Ia = I0*np.exp(-tauinf)

		#Computing emergent flux and core flux
		F1[ph] = simps(simps(Ia,XX*Rstar*Rsol),YY*Rstar*Rsol)
		F2[ph] = simps(simps(I0,XX*Rstar*Rsol),YY*Rstar*Rsol)

	#Differential magnitude from normalized flux
	dm=-2.5*np.log10(F1/F2)
	
	#t1=time.time()
	#total = t1-t0
	#print total
	return dm[1:]+dm0-dm[0]


# Stokes Q and U curve synthesis (optically thin approximation) 
#-------------------------------------------------------------------------------
def POL(phi, inc, beta, rhow, rhoc, rhoh, Rstar, Rc, QIS, UIS, thetaIS):
	
	#Defining phase grid size
	PH=len(phi)

	#Defining magnetosphere grid size
	N = np.shape(rhow)
	NNx = N[1]
	NNy = N[2]
	NNz = N[0]

	#Variable setup
	Q=np.zeros(PH) #Stokes U
	U=np.zeros(PH) #Stokes U

	#Defining spatial grids
	XX=np.linspace(-Rc,Rc,NNx)
	YY=np.linspace(-Rc,Rc,NNy)
	ZZ=np.linspace(-Rc,Rc,NNz)
	dX=np.abs(XX[0]-XX[1])
	dY=np.abs(YY[0]-YY[1])
	dZ=np.abs(ZZ[0]-ZZ[1])
	dx=dX*Rstar*Rsol
	dy=dY*Rstar*Rsol
	dz=dZ*Rstar*Rsol

	#Creating 3D meshgrids
	Z_grid, X_grid, Y_grid = np.meshgrid( ZZ, XX, YY, indexing='xy')
	R_grid = np.sqrt( Z_grid**2 + X_grid**2 + Y_grid**2 )
	P_grid = np.sqrt( Z_grid**2 + Y_grid**2 )
	MU_grid = X_grid/R_grid                 #cos(theta)
	NU_grid = np.sqrt(1. - MU_grid**2)      #sin(theta)
	CSPHI_grid = Z_grid/P_grid
	SNPHI_grid = Y_grid/P_grid
	CS2PHI_grid = CSPHI_grid**2 - SNPHI_grid**2
	SN2PHI_grid = 2.*SNPHI_grid*CSPHI_grid

	#Depolarisation factor (see Fox 1991)
	#D=np.zeros([NNz,NNx,NNy])
	D=np.sqrt(1.-1./R_grid**2)

	for ph in range(PH):

		#Trick for removing occulted regions (probably not the most efficient way )
		RHO=rhoh+rhoc+rhow	
		RHO_rot = np.zeros([NNz,NNx,NNy])
		alpha=np.arccos(csalpha(phi[ph]*2.*np.pi,np.radians(inc),np.radians(beta)))
		for k in range(0,NNx):
			RHO_rot[:,k,:] = rotate(RHO[:,k,:],np.degrees(alpha),reshape=False)
		RHO_rot[ R_grid<1.0 ] = 0.
		RHO_rot[ (np.sqrt(Z_grid**2+Y_grid**2)<1) & (X_grid<0) ] = 0 
		for k in range(0,NNx):
			RHO[:,k,:] = rotate(RHO_rot[:,k,:],np.degrees(-alpha),reshape=False)
			
		#Removing data inside star and outside closure radius
		RHO[ R_grid < 1.0 ] = 0. 
		RHO[ R_grid > Rc ] = 0.
		MU_grid[ R_grid < 1.0 ] = 0.
		MU_grid[ R_grid > Rc ] = 0. 
		D[ R_grid < 1.0 ] = 0.
		D[ R_grid > Rc ] = 0.

		#Electron density
		ne=RHO*alphae/mp

		#Computation of integral moments
		tau0 = 0.5*sigma0*simps(simps(simps(D*ne/(R_grid*Rstar*Rsol)**2,XX*Rstar*Rsol),YY*Rstar*Rsol),ZZ*Rstar*Rsol)
		taugamma0 = 0.5*sigma0*simps(simps(simps(D*ne/(R_grid*Rstar*Rsol)**2*MU_grid**2,XX*Rstar*Rsol),YY*Rstar*Rsol),ZZ*Rstar*Rsol)
		taugamma1 = 0.5*sigma0*simps(simps(simps(D*ne/(R_grid*Rstar*Rsol)**2*2*MU_grid*NU_grid*CSPHI_grid,XX*Rstar*Rsol),YY*Rstar*Rsol),ZZ*Rstar*Rsol)
		taugamma2 = 0.5*sigma0*simps(simps(simps(D*ne/(R_grid*Rstar*Rsol)**2*2*MU_grid*NU_grid*SNPHI_grid,XX*Rstar*Rsol),YY*Rstar*Rsol),ZZ*Rstar*Rsol)
		taugamma3 = 0.5*sigma0*simps(simps(simps(D*ne/(R_grid*Rstar*Rsol)**2*NU_grid**2*CS2PHI_grid,XX*Rstar*Rsol),YY*Rstar*Rsol),ZZ*Rstar*Rsol)
		taugamma4 = 0.5*sigma0*simps(simps(simps(D*ne/(R_grid*Rstar*Rsol)**2*NU_grid**2*SN2PHI_grid,XX*Rstar*Rsol),YY*Rstar*Rsol),ZZ*Rstar*Rsol)

		#Calculation of Stokes Q and U curves
		amp = (tau0-3.*taugamma0)*100.
		taugamma1 = taugamma1*100.
		taugamma2 = taugamma2*100.
		taugamma3 = taugamma3*100.
		taugamma4 = taugamma4*100.

		QU = QU_(phi[ph], amp, taugamma1, taugamma2, taugamma3, taugamma4, inc, beta, thetaIS, QIS, UIS)
		Q[ph] = QU[0] 
		U[ph] = QU[1]

	return [Q,U]


# Halpha EW curve synthesis (ignoring HeII blend)
#-------------------------------------------------------------------------------
def Halpha(phi, A, B, rhow, rhoc, rhoh, vw, vc, vh, tw, tc, th, Rstar, Rc, Teff, fcl, lda0, lda, phot, vmac, FWHM_lda):
	
	#Defining phase grid size
	PH=len(phi)

	#Defining magnetosphere grid size
	N = np.shape(rhow)
	NNx = N[1]
	NNy = N[2]
	NNz = N[0]
	Nz = 50

	#Considering density squared enhancements due to clumping
	rhow=rhow*np.sqrt(fcl)
	rhoc=rhoc*np.sqrt(fcl)
	rhoh=rhoh*np.sqrt(fcl)

	#Defining spatial grids
	XX=np.linspace(-Rc,Rc,NNx)
	YY=np.linspace(-Rc,Rc,NNy)
	ZZ=np.linspace(-Rc,Rc,NNz)
	dX=np.abs(XX[0]-XX[1])
	dY=np.abs(YY[0]-YY[1])
	dZ=np.abs(ZZ[0]-ZZ[1])
	dx=dX*Rstar*Rsol
	dy=dY*Rstar*Rsol
	dz=dZ*Rstar*Rsol

	#Creating 3D meshgrids
	X_grid, Y_grid = np.meshgrid( XX, YY, indexing='xy')
	P_grid = np.sqrt( X_grid**2 + Y_grid**2 )
	Z_grid, X_grid, Y_grid = np.meshgrid(ZZ, XX, YY, indexing='xy')
	R_grid = np.sqrt( Z_grid**2 + X_grid**2 + Y_grid**2 )
	MU_grid = X_grid / R_grid
	PHI_grid = np.arctan2(Y_grid,Z_grid)

	#Dipole cartesian unit vectors
	xhat = ( 3*MU_grid*np.sqrt(1.-MU_grid**2) )/np.sqrt(1.+3.*MU_grid**2)*np.cos(PHI_grid)
	yhat = ( 3*MU_grid*np.sqrt(1.-MU_grid**2) )/np.sqrt(1.+3.*MU_grid**2)*np.sin(PHI_grid)
	zhat = ( 3*MU_grid**2 - 1. )/np.sqrt(1.+3.*MU_grid**2) 

	#Rest frequency
	nu0 = c/lda0
	nu = c/lda

	#Setting electron and radiative temperatures
	T_e=0.75*Teff
	T_rad=0.77*Teff
	S=Bnu(nu0,T_e)/Bnu(nu0,T_rad)

	#Variable setup
	NU=len(nu)
	Pem=np.zeros([PH,NU])
	Pabs=np.zeros([PH,NU])
	P=np.zeros([PH,NU])
	W=np.zeros(PH)
	I0=np.zeros([NNx,NNy])
	I0[ P_grid<1.0 ]=1.
	for ph in range(0,PH):	
		
		alpha=np.arccos(csalpha2(phi[ph]*2.*np.pi,np.radians(A),np.radians(B)))

		RHOw_rot=np.zeros([NNz,NNx,NNy])
		RHOc_rot=np.zeros([NNz,NNx,NNy])
		RHOh_rot=np.zeros([NNz,NNx,NNy])
		Vw_rot=np.zeros([NNz,NNx,NNy])
		Vc_rot=np.zeros([NNz,NNx,NNy])
		Vh_rot=np.zeros([NNz,NNx,NNy])
		Tw_rot=np.zeros([NNz,NNx,NNy])
		Tc_rot=np.zeros([NNz,NNx,NNy])
		Th_rot=np.zeros([NNz,NNx,NNy])
		Vzhat_d = np.zeros([NNz,NNx,NNy])

		#Rotating dipole unit vector according to rotational phase
		xhat_rot = xhat
		yhat_rot = yhat*np.cos(alpha) - zhat*np.sin(alpha)
		zhat_rot = yhat*np.sin(alpha) + zhat*np.cos(alpha)

		#Trick to make dipole filed lines of opposing colatitude negative in sign 
		if alpha < np.pi/4 or alpha > 3.*np.pi/4:
			Vzhat_d[ X_grid > -np.tan(alpha)*Y_grid ] = -1.
			Vzhat_d[ X_grid < -np.tan(alpha)*Y_grid ] = 1.
		if alpha > np.pi/4 and alpha < 3.*np.pi/4:
			Vzhat_d[ (Y_grid > -(1./np.tan(alpha))*X_grid )] = -1.			
			Vzhat_d[ (Y_grid < -(1./np.tan(alpha))*X_grid) ] = 1.	

		#Line-of-sight unit vector in the rotated frame
		Vzhat_d = Vzhat_d*zhat_rot

		#Rotation of density, speed and temperature cubes 
		for k in range(0,NNx):
			RHOw_rot[:,k,:] = rotate(rhow[:,k,:],np.degrees(alpha),reshape=False,cval=0.)
			RHOc_rot[:,k,:] = rotate(rhoc[:,k,:],np.degrees(alpha),reshape=False,cval=0.)
			RHOh_rot[:,k,:] = rotate(rhoh[:,k,:],np.degrees(alpha),reshape=False,cval=0.)
			Vw_rot[:,k,:]  = rotate(vw[:,k,:],np.degrees(alpha),reshape=False,cval=0.)
			Vc_rot[:,k,:] = rotate(vc[:,k,:],np.degrees(alpha),reshape=False,cval=0.)
			Vh_rot[:,k,:] = rotate(vh[:,k,:],np.degrees(alpha),reshape=False,cval=0.)
			Tw_rot[:,k,:] = rotate(tw[:,k,:],np.degrees(alpha),reshape=False,cval=Teff)
			Tc_rot[:,k,:] = rotate(tc[:,k,:],np.degrees(alpha),reshape=False,cval=Teff)
			Th_rot[:,k,:] = rotate(th[:,k,:],np.degrees(alpha),reshape=False,cval=Teff)

		#Line-of-sight component in the rotated frame
		Vzw_rot = -Vw_rot*Vzhat_d
		Vzc_rot = Vc_rot*Vzhat_d
		Vzh_rot = -Vh_rot*Vzhat_d

		#Removing occulted regions
		RHOw_rot[ R_grid<1.0 ]=0.
		RHOw_rot[ (np.sqrt(Z_grid**2+Y_grid**2)<1) & (X_grid<0) ] = 0
		RHOc_rot[ R_grid<1.0 ]=0.
		RHOc_rot[ (np.sqrt(Z_grid**2+Y_grid**2)<1) & (X_grid<0) ] = 0
		RHOh_rot[ R_grid<1.0 ]=0.
		RHOh_rot[ (np.sqrt(Z_grid**2+Y_grid**2)<1) & (X_grid<0) ] = 0

		#Electron density
		New_rot=RHOw_rot*alphae/mp
		Nec_rot=RHOc_rot*alphae/mp
		Neh_rot=RHOh_rot*alphae/mp

		#Proton density
		Npw_rot=RHOw_rot*alphap/mp
		Npc_rot=RHOc_rot*alphap/mp
		Nph_rot=RHOh_rot*alphap/mp	

		for ll in range(0,NU):

			#Convert wavelength to velocity space
			v = (lda[ll]-lda0)/lda0*c
			vth = (2.*kb*Teff/mp)**0.5

			#Gaussian variance
			vtot = np.sqrt(vth**2 + vmac**2)
			nuD = nu0*vtot/c

			#Lorentzian FWHM
			FWHM_nu = FWHM_lda*c/lda0**2
			nuN = FWHM_nu/(2.*np.pi)

			#Doppler shifts
			x = nu[ll] - nu0
			u_w = x - nu0*Vzw_rot/c
			u_c = x - nu0*Vzc_rot/c
			u_h = x - nu0*Vzh_rot/c

			#Incremental optical depth
			dtau_w = chi32(u_w,nuD,nuN,Npw_rot,New_rot,T_e)
			dtau_c = chi32(u_c,nuD,nuN,Npc_rot,Nec_rot,T_e)
			dtau_h = 0.#chi32(u_h0,nuD,nuN,Nph_rot,Neh_rot,T_e)
			dtau = dtau_w + dtau_c + dtau_h
				
			#Optical depth
			tauinf_w = simps(dtau_w,ZZ*Rsol*Rstar,axis=0)
			tauinf_c = simps(dtau_c,ZZ*Rsol*Rstar,axis=0)
			tauinf_h = 0.#simps(dtau_h0,Z*Rsol*Rstar,axis=0)
			tauinf = trapz(dtau,ZZ*Rsol*Rstar,axis=0)
			
			#Emergent intensity and flux
			tau = cumtrapz(dtau,ZZ*Rsol*Rstar,axis=0,initial=0.)
			#print(tau[-1,:,:],tauinf)
			#print(np.shape(tau), np.shape(dtau))
			Iabs = I0*np.exp(-tauinf)*phot[ll]
			Iem = (1.-np.exp(-tauinf))*S #trapz(np.exp(-tau)*S,tau,axis=0)
			P[ph,ll] = trapz(trapz(Iabs+Iem,XX),YY)/np.pi
			#plt.imshow(np.log10(tauinf))
			#plt.show()

		#Calculation of equivalent width
		W[ph]=ew(lda*10**8,P[ph,:]/P[ph,0],lda[0]*10**8,lda[-1]*10**8,yerr='None')

	return [W,P]



