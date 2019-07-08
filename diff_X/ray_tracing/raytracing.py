import numpy as np

''' Ray tracing code for X-ray diffraction

    v2, use base change
'''


def rotation_matrix(omega, psi, phi):
    '''Rotation matrix defined with Euler's angles
        i.e. Body Rotation (intrinsic) with axis order 'yxz'
        
        angles are in radian
    '''
    s_omega, c_omega = np.sin(omega), np.cos(omega)
    s_psi, c_psi = np.sin(psi), np.cos(psi)
    s_phi, c_phi = np.sin(phi), np.cos(phi)

    R = np.array([[ s_omega*s_phi*s_psi + c_omega*c_phi,
                    s_phi*c_psi,
                   -s_omega*c_phi + s_phi*s_psi*c_omega],
                  [ s_omega*s_psi*c_phi - s_phi*c_omega,
                    c_phi*c_psi,
                    s_omega*s_phi + s_psi*c_omega*c_phi],
                  [ s_omega*c_psi,
                   -s_psi,
                    c_omega*c_psi]])
    
    return R

# test
# R = rotation_matrix(3, 2, 1)
# np.matmul(R, R.T)

def change_base(A, u, angles, offset):
    """Change the reference frame of the given rays (A, u)
       
       angles = (omega, psi, phi) in radian
       offset = (x0, y0, z0) in mm
    """
    omega, psi, phi = angles
    offset = np.asarray(offset)

    R = rotation_matrix(omega, psi, phi)

    A_prime = np.matmul(A, R.T) - offset
    u_prime = np.matmul(u, R.T)
    
    return A_prime, u_prime

# test
#u = np.array([[-1, 0, 0], [1, 1, 1]])
#A = np.array([[1, 0, 0], [1, 0, 0]])

#change_base(A, u, (np.pi/2, 0, 0), (0, 0, -1))


def rectangle_intersection(A, u, angles, offset, width, height):
    '''Projection of the ray (A, u) on the plane xy
        of the rotated (angles) and translated (offset) base
        
        Returns
        -------
        lost: flag indicating if ray go through the rectangle
        in_plane_uv: coordinates of the collision point in the plane ref. base
        B: coordinates of the collision in the incident ref. base
    '''
    A_prime, u_prime = change_base(A, u, angles, offset)

    time_to_plane = -np.divide(A_prime[:, 2], u_prime[:, 2],
                               where=u_prime[:, 2] < 0)

    B = A + time_to_plane[:, np.newaxis]*u  # point of intersection in the laboratory frame

    in_plane_uv = A_prime[:, 0:2] + time_to_plane[:, np.newaxis]*u_prime[:, 0:2]


    lost = time_to_plane < 0
    lost = np.logical_or(lost, np.abs(in_plane_uv[:, 0]) > width/2 )
    lost = np.logical_or(lost, np.abs(in_plane_uv[:, 1]) > height/2 )

    #in_plane_uv[mask, :] = np.NaN
    
    return lost, in_plane_uv, B


def normalize(array_of_vector):
    """
    Normalize the given array of vectors.

    Parameters
    ----------
    array_of_vector : (N, dim) array
        Vector along the last dim

    Returns
    -------
    y : array (N, dim)

    """
    array_of_vector = np.asarray(array_of_vector)
    norm = np.linalg.norm(array_of_vector, axis=-1, keepdims=True)

    return np.divide(array_of_vector, norm, where=norm > 0)


# Elements and optics

def source(N, width, height,
           divergence_z, divergence_y, position):
    """
    Generate random rays from a rectangular source 
    with gaussian distributed divergence

    Parameters
    ----------
    N : integer
        Number of generated rays
    width, height : floats [mm]
        dimension of the rectangular source, in mm
    divergence_z, divergence_y : floats [degree]
        Angle of beam divergence
        (i.e. standard diviation of a gaussian distribution)
    position : float [mm]
        Source position along global X axis
    
    Returns
    -------
    A, u : (N, 3) float arrays
        Position and direction of rays

    """
        
    # Directions:
    ux = -np.ones((N, 1))
    uy = np.random.randn(N, 1)* divergence_y *np.pi/180
    uz = np.random.randn(N, 1)* divergence_z *np.pi/180
    u = np.hstack([ux, uy, uz])
    u = normalize(u)
    
    # Beam shape
    Ax = np.ones((N, 1)) * position
    A_height = (np.random.rand(N, 1) - 0.5) * height  # mm, square, along Y
    A_width = (np.random.rand(N, 1) - 0.5) * width  # mm, square, along Z
    A = np.hstack([Ax, A_height, A_width])

    return A, u


def planar_powder(A, u,
                  omega, psi, phi, X, Y, Z,
                  width, height,
                  gamma_range, deuxtheta_diff):
    """
    Diffraction by a perfect planar powder
        
    Parameters
    ----------
    A, u : (N, 3) arrays
        Incident rays
    omega, psi, phi: floats [degree]
        Sample holder orientation
    X, Y, Z : floats [mm]
        Sample holder position
    width, height: floats [mm]
        Dimension of the rectangular sample
    gamma_range : float [degree]
        Max-min out-of-plane diffracted angle
        Angles are generated using uniform random distribution
    deuxtheta_diff : float 
        Diffracted angle (Bragg's law)

    Returns
    -------
    B, d : (N, 3) arrays
        Diffrated rays
    through : (N, ) boolean array
        True if the ray was inside the sample
    uv : (N, 2) float array
        Intersection points in the sample plane 

    Notes
    -----
    The orientation for gamma=0 is defined as perpendicular
    to the vertical direction (0, 1, 0) and righ-handed rotation with u

    """
    deuxtheta = np.pi/180 * deuxtheta_diff
    N = np.shape(A)[0]
    gamma = gamma_range*(np.random.rand(N,) - 0.5) *np.pi/180
    
    # Intersection with the sample
    angles = np.pi/180 * np.array([omega, psi, phi])
    offset = np.array([X, Y, Z])
    lost, uv, B = rectangle_intersection(A, u,
                                         angles, offset, width, height)

    through = ~lost
    # Diffracted cone
    ref_plane_normal = np.array((.0, 1., .0))

    u = normalize(u)
    ref_plane_normal = normalize(ref_plane_normal)

    # Define a new base with u
    gamma_zero = normalize(np.cross(ref_plane_normal, u))
    gamma_90 = normalize(np.cross(u, gamma_zero))

    # diffracted direction:
    d = u*np.cos(deuxtheta) + np.sin(deuxtheta)*(gamma_zero*np.cos(gamma)[:, np.newaxis] + \
                               gamma_90*np.sin(gamma)[:, np.newaxis])
    
    return B, d, through, uv


def slit_detector(A, u, deuxtheta,
                  distance, offset,
                  width, height):
    """
    Simple slit detector
    
    Parameters
    ----------
    A, u : (N, 3) arrays
        Incident rays
    deuxtheta : float [degree]
        Angular position of the detector
    distance : float [mm]
        Distance from gonio center to the detector plane
    offset : float [mm]
        Position of the slit center relative to the detector position
    width, height : floats [mm]
        Dimensions of the receveing slit

    Returns
    -------
    through : (N, ) boolean array
        True if ray is detected (i.e. go through the slit)
    uv_detector : (N, 2) float array
        Intersection points in the detector plane 

    """


    angles = (deuxtheta*np.pi/180 + np.pi/2, 0, 0) # i.e. omega, phi, psi
    offset = (offset, 0, -distance)

    lost_detect, uv_detector, _ = rectangle_intersection(A, u,
                                                         angles, offset,
                                                         width, height)
    through = ~lost_detect
    return through, uv_detector


def plate_collim_detector(A, u, deuxtheta,
                          distance, offset,
                          width, height,
                          length, nbr_plates, acceptance):
    """
    Plate collimator detector

    Parameters
    ----------
    A, u : (N, 3) arrays
        Incident rays
    deuxtheta : float [degree]
        Angular position of the detector
    distance : float [mm]
        Distance from gonio center to the detector front plane
    offset : float [mm]
        Position of the slit center relative to the detector position
    width, height : floats [mm]
        Overall dimensions of the receveing slit
    length : float [mm]
        Length of the plates
    nbr_plates : integer
        number of plates, used to estimate the thickness of the plates
    acceptance : float [degree]
        half-angle
        used to estimate the gap between the plates
        tan( acceptance ) = gap/length
        
    Returns
    -------
    through : (N, ) boolean array
        True if ray is detected (i.e. go through the slit)
    uv_detector : (N, 2) float array
        Intersection points in the detector plane 

    """
    gap_width = np.tan(acceptance *np.pi/180)*length
    period = width/(nbr_plates + 1)
    
    offset = (offset, 0, -distance)
    angles = (deuxtheta*np.pi/180 + np.pi/2, 0, 0)
    A_prime, u_prime = change_base(A, u, angles, offset)

    time_to_front_plane = -np.divide(A_prime[:, 2], u_prime[:, 2],
                               where=u_prime[:, 2] < 0)

    time_to_back_plane = -np.divide(A_prime[:, 2] - length, u_prime[:, 2],
                               where=u_prime[:, 2] < 0)
    
    front_plane_uv = A_prime[:, 0:2] + time_to_front_plane[:, np.newaxis]*u_prime[:, 0:2]
    back_plane_uv = A_prime[:, 0:2] + time_to_back_plane[:, np.newaxis]*u_prime[:, 0:2]

    #through = time_to_front_plane > 0
    through = np.zeros((A.shape[0],), dtype=bool)
    for k in range(nbr_plates+1):
        x_center = -width/2 + k*period
        
        through_front = np.abs(front_plane_uv[:, 0]-x_center) < gap_width/2
        through_back =  np.abs(back_plane_uv[:, 0]-x_center) < gap_width/2
        through_k = np.logical_and(through_front, through_back)
        
        through = np.logical_or(through, through_k)
        
    # vertical    
    through = np.logical_and(through, np.abs(front_plane_uv[:, 1]) < height/2 )
    through = np.logical_and(through, np.abs(back_plane_uv[:, 1]) < height/2 )
    return through, front_plane_uv