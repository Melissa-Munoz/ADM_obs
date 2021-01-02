import numpy as np
import matplotlib.pyplot as plt
from ADM import ADM
import obs
import time

t0 = time.time()

#Physical properties of the star
#-------------------------------------------------------------------------------
#Effective temperature, Teff, is in Kelvins
#Stellar mass, Mstar, is in solar mass
#Stellar radius, Rstar, is in solar radius
#Terminal velocity, Vinf, is in km/s
#Mass-loss rate, Mdot, is in solar mass per year
#Polar field strength, Bstar is in Gauss
Teff = 35000.0 
Mstar = 30.0
Rstar = 15.
Vinf = 2500.0
Mdot = 10**(-6.0)
Bstar = 2500.0

#Geometric angles
#-------------------------------------------------------------------------------
#inclination angle, inc, in degrees
#magnetic obliquity, beta, in degrees
inc = 45.
beta = 45.
A = inc+beta
B = np.abs(inc-beta)

#Extra paramaters
#-------------------------------------------------------------------------------
#Smoothing length, delta
#Vertical offset in differentlia magnitde, dm0 (constant)
delta = 0.1 
dm0 = 0
QIS = 0
UIS = 0
thetaIS = 0
vmac = 100.0
FWHM_lda = 0.0
fcl = 5.0
lda0 = 6562.8
dlda = 15.

#Calling ADM
#-------------------------------------------------------------------------------
phi = np.linspace(0.,1.,25) #rotational phase
lda = np.linspace(lda0-dlda,lda0+dlda,15)
phot=np.ones(len(lda)) #photospheric line profile (ignored here)
Nx = Ny = Nz = 50 #grid size 
out = ADM(Nx, Ny, Nz, Teff, Mstar, Rstar, Vinf, Mdot, Bstar, delta)
rhow = out[0]
rhoc = out[1]
rhoh = out[2]
vw = out[3]
vc = out[4]
vh = out[5]
tw = out[6]
tc = out[7]
th = out[8]
Rc = out[9]
modelLC = obs.LC(phi, A, B, rhow, rhoc, rhoh, Rstar, Rc, dm0 )
modelQU = obs.POL(phi, inc, beta, rhow, rhoc, rhoh, Rstar, Rc, QIS, UIS, thetaIS)
modelEW = obs.Halpha(phi, A, B, rhow, rhoc, rhoh, vw, vc, vh, tw, tc, th, Rstar, Rc, Teff, fcl, lda0*10**(-8), lda*10**(-8), phot, vmac*100000., FWHM_lda**10**(-8))

#Plotting phased observable quantities
#-------------------------------------------------------------------------------
plt.figure(figsize=(9,6))
plt.plot(phi,modelLC,'k')
plt.plot(phi+1,modelLC,'k')
plt.plot(phi-1,modelLC,'k')
plt.gca().invert_yaxis()
plt.xlabel('Rotational phase',fontsize=14)
plt.ylabel('Differential magnitude [mag]',fontsize=14)
plt.xlim([-0.5,1.5])
plt.show()

fig, ax = plt.subplots(2, figsize=(9,6),sharex=True,sharey=True)
ax[0].plot(phi, modelQU[0],'k')
ax[0].plot(phi+1, modelQU[0],'k')
ax[0].plot(phi-1, modelQU[0],'k')
ax[1].plot(phi, modelQU[1],'k')
ax[1].plot(phi+1, modelQU[1],'k')
ax[1].plot(phi-1, modelQU[1],'k')
ax[0].set_ylabel('Q [%]',fontsize=14)
ax[1].set_ylabel('U [%]',fontsize=14)
ax[1].set_xlabel('Rotational phase',fontsize=14)
#ax[0].set_xlim([-0.5,1.5])
ax[1].set_xlim([-0.5,1.5])
plt.show()

plt.figure(figsize=(6,6))
plt.plot(modelQU[0],modelQU[1],'k')
plt.ylabel('Q [%]',fontsize=14)
plt.xlabel('U [%]',fontsize=14)
plt.show()

plt.figure(figsize=(9,6))
plt.plot(phi,modelEW[0],'k')
plt.plot(phi+1,modelEW[0],'k')
plt.plot(phi-1,modelEW[0],'k')
plt.xlabel('Rotational phase',fontsize=14)
plt.ylabel(r'Equivalent width [$\AA$]',fontsize=14)
plt.xlim([-0.5,1.5])
plt.show()

plt.figure(figsize=(6,6))
for i in range(len(phi)):
	plt.plot(lda,modelEW[1][i,:])
plt.xlabel('Wavelength [$\AA$]',fontsize=14)
plt.ylabel('Normalized flux',fontsize=14)
plt.xlim([lda0-10,lda0+10])
plt.show()

#Total time (to be improved in a future update)
t1=time.time()
total = t1-t0
print('time',total)
