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
    """ Normalize the given array of vectors.

        Parameters
        ----------
        array_of_vector : (N, dim) array
            vector along the last dim

        Returns
        -------
        y : array (N, dim)
    """
    array_of_vector = np.asarray(array_of_vector)
    norm = np.linalg.norm(array_of_vector, axis=-1, keepdims=True)

    return np.divide(array_of_vector, norm, where=norm > 0)


def source(N, source_width, source_height,
           divergence_z, divergence_y, source_position):
    """Generate rays from a rectangular source 
    
    Parameters
    ----------
    N: number of generated rays
    width, height: dimension of the rectangular source, in mm
    divergence_z, divergence_y: degree, angle of beam divergence (std of gaussian distribution)
    source_position: source position along global X axis
    
    Returns
    -------
    A, u: position and direction (N, 3) arrays
    """
        
    # Directions:
    ux = -np.ones((N, 1))
    uy = np.random.randn(N, 1)* divergence_y*np.pi/180
    uz = np.random.randn(N, 1)* divergence_z*np.pi/180
    u = np.hstack([ux, uy, uz])
    u = normalize(u)
    
    # Beam shape
    Ax = np.ones((N, 1)) * source_position
    A_height = (np.random.rand(N, 1) - 0.5) * source_height  # mm, square, along Y
    A_width = (np.random.rand(N, 1) - 0.5) * source_width  # mm, square, along Z
    A = np.hstack([Ax, A_height, A_width])

    return A, u


# Diffracted directions
def planar_powder(A, u,
                  omega, psi, phi, X, Y, Z, sample_width, sample_height,
                  gamma_range, deuxtheta_diff):
    '''Diffraction by a perfect planar powder
        
    Parameters
    ----------
    A, u: (N, dim) array
        incident rays
    omega, psi, phi: floats
        sample holder orientation (degree)
    X, Y, Z: floats
        sample holder position (mm)
    sample_width, sample_height: floats
        dimension of the rectangular sample
    gamma_range: float (degree)
        max-min out of plane diffracted angle
    deuxtheta_diff: float
        diffracted angle (Bragg's law) (degree)
    '''
    deuxtheta = np.pi/180 * deuxtheta_diff
    N = np.shape(A)[0]
    gamma = gamma_range * np.pi/180*(np.random.rand(N,) - 0.5)
    
    # Intersection with the sample
    angles = np.pi/180 * np.array([omega, psi, phi])
    offset = np.array([X, Y, Z])
    mask, uv, B = rectangle_intersection(A, u,
                                         angles, offset, sample_width, sample_height)

    
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
    
    return B, d, mask, uv


def slit_detector(A, u, deuxtheta,
                  detector_distance, detector_offset, detector_slit_angle,
                  slit_conversion_distance, detector_height):
    """ Simple slit detector
    """
    width = slit_conversion_distance*detector_slit_angle *np.pi/180 # mm
    height = detector_height

    angles = (deuxtheta*np.pi/180 + np.pi/2, 0, 0) # i.e. omega, phi, psi
    offset = (detector_offset, 0, -detector_distance)

    lost_detect, uv_detector, C = rectangle_intersection(A, u,
                                                         angles, offset,
                                                         width, height)

    return ~lost_detect, uv_detector


def plate_collim_detector(A, u, deuxtheta,
                          detector_offset, detector_distance,
                         length, detector_width, detector_height,
                         nbr_plates, detector_acceptance):
    '''Plate collimator detector
        
        acceptance in degree, half-angle
        
        origin is defined as center of the front surface
        
          'deuxtheta':0, # deux-theta, deg
    'detector_distance':280, # mm, distance from gonio center to receving slit
    'detector_offset':0, # mm, offset along Z
    'length':96, # mm
    'detector_width':22,
    'detector_height':20,
    'nbr_plates':39,
    'detector_acceptance':0.27 # degree
    '''
    gap_width = np.tan(detector_acceptance *np.pi/180)*length
    period = detector_width/(nbr_plates + 1)
    
    offset = (detector_offset, 0, -detector_distance)
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
        x_center = -detector_width/2 + k*period
        
        through_front = np.abs(front_plane_uv[:, 0]-x_center) < gap_width/2
        through_back =  np.abs(back_plane_uv[:, 0]-x_center) < gap_width/2
        through_k = np.logical_and(through_front, through_back)
        
        through = np.logical_or(through, through_k)
        
    # vertical    
    through = np.logical_and(through, np.abs(front_plane_uv[:, 1]) < detector_height/2 )
    through = np.logical_and(through, np.abs(back_plane_uv[:, 1]) < detector_height/2 )
    return through, front_plane_uv