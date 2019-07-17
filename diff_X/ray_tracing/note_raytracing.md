Ray Tracing
===========

## Reference Frame

Laboratory: (X, Y, Z)
center at the rotation center
X toward the X-Ray source 
Y is vertical
Z toward the doors


Sample rotation angles:
omega around Y --> (x, Y, z)
psi around x --> (x, y', z')
phi around z' = n --> (u, v, w)



I(X,Y,Z,phi,psi,omega,2theta) = integrale sur
- rho(x, y, ux, uy, lambda) source
- gamma: diffracted direction (+ par exemple diffracted depth)



vegas algo


rho(x, y, ux, uy, lambda)*Transmission(x, y, ux, uy, lambda, gamma)


rho = rho_y*rho_z*rho_lambda*rho_uy*rho_uz


ray: (lambda, A, u)

Rectangle/fente:
- (width, height), position X=(x,y,z), normal n=(u,v,w)
- (width, height), (omega, psi, phi)=rotation around (Y, x, z) resp., delta_x,y,z offsets 

- input: A, u
- output: mask=1 or 0, B lab.frame coords, uv  inplane coords


## Transmission

Parameters:
 - incident ray: A, u, lambda
 - sample: omaga, phi, psi, offset, height, width
 - diffraction: d_hkl, gamma (relative to lab. XZ plane)
 - detecteur: position 2theta, +geometry
 
 
Returns:
  - 0 or 1


Steps:
1. on sample?  yes/no
    -> diffraction at angle gamma: new ray (B, u')
2. through detector?
    2.1 on detector?
    2.i through slits #i



    
    

# Numba

no np.matmul in numba
np.matmul faster than manual implementation




Calibration
===========

source: w, h profiles, and divergence
detecteur fente 1/4°

Faisceau direct--> 


Exp:
Source





# Plot

https://stackoverflow.com/questions/42639129/is-matplotlib-scatter-plot-slow-for-large-number-of-data




# diffract

    # Parameters
     - Incident beam, source & optique
         A: beam size --> distribution
         u: divergence --> distribution
         lambda: energy --> distribution
     - Sample (perfect powder)
         size: height, width
         d(hkl) --> distribution / discrete
         (absorption, grain size, monocristal...)
     - Diffraction
         gamma (relative to lab. XZ plane) --> distribution
     - Detecteur
         geometry: slit width and height, offset

    # Gonio movements, scan, measure:
     - sample stage: omega, phi, psi, X, Y, Z
     - detecteur position: 2theta
     
     # Variables

- x, y, source
     - 