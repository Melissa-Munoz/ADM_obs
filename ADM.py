#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ADM-based Stokes Q ans U curves synthesis 
# Melissa Munoz
# Updated Dec 2020
#
# See publication Munoz et al. 2020, in prep
# See also Owocki et al. 2006 for more details on the ADM formalism
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#-------------------------------------------------------------------------------
# Library import ---------------------------------------------------------------
#-------------------------------------------------------------------------------

import numpy as np
from scipy.optimize import newton
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time 


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
# ADM auxiliary equations ------------------------------------------------------
#-------------------------------------------------------------------------------

#Dipole magnetic field
def Bd(r,mu,mustar):
	return (1./r)**3*((1+3*mu**2)/(1+3*mustar**2))**0.5

# Wind upflow
#-------------------------------------------------------------------------------

def w(r):
	return 1.-1./r

def vw(r,vinf):
	return vinf*(1.-1./r)

def rhow(r,mu):
	return 2.*np.sqrt(r - 1. + mu**2)*np.sqrt(1.+3.*mu**2)/((r- 1.)*(4.*r - 3. + 3.*mu**2))*(1./r)**(3./2.)


# Hot post-shock 
#-------------------------------------------------------------------------------

def wh(r,rs,mu,mus,Tinf,Teff):
	ws=w(rs)
	Ts=Tinf*ws**2
	return ws/4.*(Th(rs,mu,mus,Tinf,Teff)/Ts)*(Bd(r,mu,mus))

def vh(r,rs,mu,mus,Tinf,Teff,vinf):
	ws=w(rs)
	Ts=Tinf*ws**2
	return ws*vinf/4.*Th(rs,mu,mus,Tinf,Teff)/Ts*np.sqrt((1.+3.*mu**2)/(1.+3.*mus**2))*(rs/r)**3

def g(mu):
	return np.abs(mu - mu**3 + 3.*mu**5/5. - mu**7/7.)

def TTh(rs,mu,mus,Tinf):
	ws=w(rs)
	Ts=Tinf*ws**2
	return Ts*(g(mu)/g(mus))**(1./3.)

def Th(rs,mu,mus,Tinf,Teff):
	return np.maximum(TTh(rs,mu,mus,Tinf),Teff)

def rhoh(r,rs,mu,mus,Tinf,Teff):
	ws=w(rs)
	Ts=Tinf*ws**2
	return 4.*rhow(rs,mus)*Ts/Th(rs,mu,mus,Tinf,Teff)

# Cooled downflow 
#-------------------------------------------------------------------------------

def wc(r,mu):
	return np.abs(mu)*np.sqrt(1./r)

def vc(r,mu,ve):
	return np.abs(mu)*np.sqrt(1./r)*ve

def rhoc(r,mu,delta):
	return 2.*np.sqrt(r - 1. + mu**2)*np.sqrt(1.+3.*mu**2)/(np.sqrt(mu**2+delta**2/r**2)*(4.*r - 3. + 3.*mu**2))*(1./r)**(2.)

def f(mus,mustar,chiinf):
	rm = 1./(1.-mustar**2)
	rs = rm*(1.-mus**2)
	ws = w(rs)
	ggmus = chiinf/(6.*mustar)*(1.+3.*mustar**2)/(1.+3.*mus**2)*(ws*rs/rm)**4*(rs)**2
	gmus = g(mus)
	return (gmus-ggmus) 



#-------------------------------------------------------------------------------
# ADM --------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def ADM(Nx, Ny, Nz, Teff, Mstar, Rstar, Vinf, Mdot, Bd, delta):

	#Defining magnetosphere grid size	
	NNx=2*Nx
	NNy=2*Ny
	NNz=2*Nz

	#Conversion of stellar properties into cgs 
	mdot=Mdot*Msol/(365.*24*3600.)
	vinf=Vinf*100000.
	rstar=Rstar*Rsol
	mstar=Mstar*Msol
	ve = np.sqrt(2.*G*mstar/rstar)
	Ve = np.sqrt(2.*G*mstar/rstar)/100000.
	rhowstar = mdot/(4.*np.pi*rstar**2*vinf)
	rhocstar = rhowstar*vinf/ve

	#Some scalling relations. see Owocki 2016
	chiinf = 0.034*(vinf/10.**8)**4*(rstar/10**12)/(Mdot/10**(-6))
	Tinf = 14*10**6*(vinf/10.**8)**2

	#Computing the Alfven radius and closure radius
	Beq=Bd/2.
	eta=(Beq)**2*rstar**2/(mdot*vinf)
	RA=0.3+(eta+0.25)**(0.25)
	Rc = RA # This can be changed occording to the user

	#Defining spatial grids
	XX=np.linspace(-Rc,Rc,NNx)
	YY=np.linspace(-Rc,Rc,NNy)
	ZZ=np.linspace(-Rc,Rc,NNz)	
	X=XX[Nx:NNx]
	Y=YY[Ny:NNy]
	Z=ZZ[Nz:NNz]
	
	#Defining density, speed, and temperature grids of each component
	Rhoh=np.zeros([Nz,Nx,Ny])
	Rhow=np.zeros([Nz,Nx,Ny])
	Rhoc=np.zeros([Nz,Nx,Ny])
	Vh=np.zeros([Nz,Nx,Ny])
	Vw=np.zeros([Nz,Nx,Ny])
	Vc=np.zeros([Nz,Nx,Ny])
	tw = np.zeros([Nz,Nx,Ny])
	tc = np.zeros([Nz,Nx,Ny])
	th = np.zeros([Nz,Nx,Ny])

	#Last closed loop
	mustar_RA = np.sqrt(1.-1./RA)
	mustars_RA = np.linspace(0.01,mustar_RA,Nz)
	mus_RA = np.zeros(Nz)
	rs_RA = np.zeros(Nz)
	r_RA = (1.-mustars_RA**2)/(1-mustar_RA**2)
	for i in range(Nz):
		try:
			tempmus = newton(f, 0.3, args=(mustars_RA[i],chiinf,))
		except RuntimeError:	
			tempmus=0.
			#print 'error LC'
		mus_RA[i]=np.abs(tempmus)
		rs_RA[i]=(1.-mus_RA[i]**2)/(1.-mustars_RA[i]**2)
	fs=interp1d( mustars_RA,mus_RA,bounds_error=False, fill_value=0. )

	#Compute ADM in first octant of the magnetosphere
	#Velocity and temperature calculations are commented out because they are not required for the light curve synthesis
	for i in range(0,Nx):
		for j in range(0,Ny):
			p=np.sqrt(X[i]**2+Y[j]**2)
			for k in range(0,Nz):
				r=np.sqrt(p**2+Z[k]**2)
				mu=Z[k]/r
				rRA=(1.-mu**2)/(1-mustar_RA**2)
				if r > 1.05:
					mustar=np.sqrt(1.-(1.-mu**2)/r)
					rm = 1./(1.-mustar**2)
					mus=fs(mustar)
					'''
					try:
						tempmus = newton(f, 0.3, args=(mustar,chiinf,))
					except RuntimeError:	
						tempmus=0.
						#print 'error LC'
					mus = np.abs(tempmus)
					'''
					rs = rm*(1.-mus**2)
					Rhow[k,i,j]=rhow(r,mu)*rhowstar
					Vw[k,i,j]=w(r)*vinf
					tw[k,i,j]=Teff		
					if r < rRA:
						Rhoc[k,i,j]=rhoc(r,mu,delta)*rhocstar
						Vc[k,i,j]=wc(r,mu)*ve
						tc[k,i,j]=Teff
						if r > rs and rs > 1.05 :
							Rhoh[k,i,j]=rhoh(r,rs,mu,mus,Tinf,Teff)*rhowstar
							Vh[k,i,j]=wh(r,rs,mu,mus,Tinf,Teff)*vinf
							th[k,i,j]=Th(rs,mu,mus,Tinf,Teff)
	
	#Transposing density in remaining octants (axial symmetry)
	Rhoh=np.concatenate([Rhoh[::-1,:,:],Rhoh],axis=0)
	Rhoh=np.concatenate([Rhoh[:,::-1,:],Rhoh],axis=1)
	Rhoh=np.concatenate([Rhoh[:,:,::-1],Rhoh],axis=2)
	Rhoc=np.concatenate([Rhoc[::-1,:,:],Rhoc],axis=0)
	Rhoc=np.concatenate([Rhoc[:,::-1,:],Rhoc],axis=1)
	Rhoc=np.concatenate([Rhoc[:,:,::-1],Rhoc],axis=2)
	Rhow=np.concatenate([Rhow[::-1,:,:],Rhow],axis=0)
	Rhow=np.concatenate([Rhow[:,::-1,:],Rhow],axis=1)
	Rhow=np.concatenate([Rhow[:,:,::-1],Rhow],axis=2)	
	
	#Transposing speed in remaining octants (axial symmetry)
	Vw=np.concatenate([Vw[::-1,:,:],Vw],axis=0)
	Vw=np.concatenate([Vw[:,::-1,:],Vw],axis=1)
	Vw=np.concatenate([Vw[:,:,::-1],Vw],axis=2)
	Vc=np.concatenate([Vc[::-1,:,:],Vc],axis=0)
	Vc=np.concatenate([Vc[:,::-1,:],Vc],axis=1)
	Vc=np.concatenate([Vc[:,:,::-1],Vc],axis=2)
	Vh=np.concatenate([Vh[::-1,:,:],Vh],axis=0)
	Vh=np.concatenate([Vh[:,::-1,:],Vh],axis=1)
	Vh=np.concatenate([Vh[:,:,::-1],Vh],axis=2)	

	#Transposing temperature in remaining octants (axial symmetry)	
	tw=np.concatenate([tw[::-1,:,:],tw],axis=0)
	tw=np.concatenate([tw[:,::-1,:],tw],axis=1)
	tw=np.concatenate([tw[:,:,::-1],tw],axis=2)
	tc=np.concatenate([tc[::-1,:,:],tc],axis=0)
	tc=np.concatenate([tc[:,::-1,:],tc],axis=1)
	tc=np.concatenate([tc[:,:,::-1],tc],axis=2)
	th=np.concatenate([th[::-1,:,:],th],axis=0)
	th=np.concatenate([th[:,::-1,:],th],axis=1)
	th=np.concatenate([th[:,:,::-1],th],axis=2)

	return [Rhow, Rhoc, Rhoh, Vw, Vc, Vh, tw, tc, th, Rc]	







