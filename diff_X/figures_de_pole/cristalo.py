from itertools import permutations, product
import numpy as np


def equivalent_directions(hkl):
    '''List the equivalent directionfor a cube
    
        hkl: tuple of Miller indices'''
    perms = permutations(hkl)
    sign = product([1, -1], [-1, 1], [-1, 1])
    dirs = product(perms, sign)
    dirs = {tuple(u*s for u, s in zip(xys, signs)) for xys, signs in dirs} 
    return dirs


# Vector based geometry:
nbr_digit = 6

def cross_product(a, b):
    ab = (a[1]*b[2] - a[2]*b[1],
          a[2]*b[0] - a[0]*b[2],
          a[0]*b[1] - a[1]*b[0])
    return np.asarray(ab)

def dot_product(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.round(np.sum(a*b), decimals=nbr_digit)

def norm(a):
    return np.sqrt(dot_product(a, a))

def norm_it(a):
    a = np.asarray(a)
    return np.copy(a) / norm(a)

def angle(a, b):
    '''Angle between vectores a and b in degrees'''
    cos_ab = dot_product(a, b)/(norm(a)*norm(b))
    cos_ab = np.round(cos_ab, decimals=nbr_digit)
    return np.arccos(cos_ab) * 180/np.pi

def plan_projection(a, n):
    ''' Project the vector `a` in the plan of normal `n`
    '''
    n = norm_it(n)
    scale_n = dot_product(a, n)
    a_plan = tuple(ai - scale_n*ni for ai, ni in zip(a, n))
    return round_it(a_plan)

def round_it(a): 
    return tuple(round(ai, nbr_digit) for ai in a)




def get_phi_psi(u, n, phi0):
    ''' Return the phi and psi angles for the given u vector (h, k, l) in degree
    
        n: normal to the surface tuple(h, k, l)
        phi0: direction corresponding to phi0. Perdendicular to n.
    '''
    n = norm_it(n)
    phi0 = norm_it(phi0)
    
    if dot_product(n, phi0) != 0:
        raise NameError('phi0 not perpendicular to n')
    
    phi90 = cross_product(phi0, n)
    
    psi = angle(u, n)
    
    if abs(psi) < 1e-4 or abs(psi-180) < 1e-4: 
        return 0, psi
    
    u_plan = norm_it( plan_projection(u, n) )

    cos = dot_product(phi0, u_plan)
    sin = dot_product(phi90, u_plan)

    phi = np.arctan2(sin, cos)*180/np.pi
    
    return phi, psi



def rodrigues_rotation(v, k, theta):
    ''' Generic rotation using the Rodrigues' rotation formula
         https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        v: the vector to rotate
        k: the axis of rotation (normed before computation)
        theta: angle of rotation in radian
    '''
        
    v, k = np.asarray(v), np.asarray(k)
    k = k / np.linalg.norm(k)
    
    return v * np.cos(theta) + np.cross(k, v)*np.sin(theta) \
            + k * np.inner(k, v)*(1 - np.cos(theta))   

# test
# rodrigues_rotation([1, 0, 0], [1, 1, 1], np.pi/180*120)