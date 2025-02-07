import numpy as np
import matplotlib.pyplot as plt
import scipy
# Define theta and phi
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi,200)

# Create Meshgrid
Theta, Phi = np.meshgrid(theta,phi)

# Cart. to Pol.
R = Theta
X = R*np.cos(Phi)
Y = R*np.sin(Phi)

# Initialize spher. harm.
Y_lm = lambda l,m: scipy.special.sph_harm(m, l, Phi, Theta).real
Y_lm_norm = Y_lm(l,m)/np.max(np.abs(Y_lm(l,m)))
l, m = 2,2

# plot
plt.figure(figsize=(8,6))
plt.pcolormesh(X, Y, Y_lm_norm,cmap='coolwarm', shading='auto')

plt.colorbar(label="Spherical Harmonic Value")
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Spherical Harmonic Y({l},{m}) in Polar Grid")

plt.axis("equal")
plt.show()