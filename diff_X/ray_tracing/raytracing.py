""" Ray Tracing

"""
import numpy as np


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


# test
assert np.allclose(normalize(np.array((2, 0, 0))),
                   np.array((1, 0, 0)))
assert np.allclose(normalize(np.array([(0, 2, 0), (0, -3, 0), (0, 0, 0), (-1, 0, 0)])),
                   np.array([(0, 1, 0), (0, -1, 0), (0, 0, 0), (-1, 0, 0)]))


def plane_intersection(A, u, plane_center, plane_normal):
    """ Intersection points of incident rays with a plane.

    returns NaN if no intersection (i.e. when opposite direction or parallel)

    Parameters
    ----------
    A : (N, dim) array
        starting points of the incident rays
    u : (N, dim) array
        directions of the incident rays
    plane_center : (dim, ) vector
        position of the intersection plane
    plane_normal : (dim, ) vector
        normal of the intersection plane

    Returns
    -------
    B : (N, dim) array
        points of intersection for each rays

    """
    # Intersection of incident beams with sample plane:
    A = np.asarray(A, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)

    AP_dot_n = np.inner((A - plane_center), plane_normal)
    u_dot_n = np.inner(u, plane_normal)

    t = - np.divide(AP_dot_n, u_dot_n,
                    where=np.logical_not(np.isclose(u_dot_n, 0)))
    t[t < 0] = np.NaN
    B = A + u * t[:, np.newaxis]

    return B


# test
A_test = np.array([(1, 0, 0), (2, 1, 0), (1, 0, 1), (1, 0, 0),
              (-1, 0, 0), (-1, 0, 0), (2, 0, 1), (2, 2, -2)])
u_test = np.array([(-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (-1, 0, -1),
              (-1, 0, 0), (1, 0, 1), (1, 0, 0), (1, 0, 0)])

plane_center_test = np.array((0, 0, 0))
plane_normal_test = np.array((1, 0, 1))

assert np.allclose(plane_intersection(A_test, u_test, plane_center_test, plane_normal_test),
                   np.array([[0., 0., 0.],
                             [0., 1., 0.],
                             [-1., 0., 1.],
                             [0.5, 0., -0.5],
                             [np.nan, np.nan, np.nan],
                             [-0.5, 0., 0.5],
                             [np.nan, np.nan, np.nan],
                             [2., 2., -2.]]), equal_nan=True)


def projection_detector_plane(A, u, detector_position, detector_normal, detector_vertical):
    """Projection on a plane, returns coordinates in the local in-plane base

    Parameters
    ----------
    A : (N, dim) array
        starting points of the incident rays
    u : (N, dim) array
        directions of the incident rays
    detector_position : (dim, ) vector
        position of the intersection plane
    detector_normal : (dim, ) vector
        normal of the intersection plane
    detector_vertical : (dim, ) vector
        direction indication for the vertical axis of the in-plane base

    Returns
    -------
    detector_u, detector_v : (N, dim) arrays
        coordinates of intersection points in the in-plane base

    """
    detector_vertical = normalize(detector_vertical)
    detector_normal = normalize(detector_normal)

    detector_horizontal = np.cross(detector_normal, detector_vertical)

    P = plane_intersection(A, u, detector_position, detector_normal)

    P_prime = P - detector_position
    detector_v = np.dot(P_prime, detector_vertical)
    detector_u = np.dot(P_prime, detector_horizontal)

    return detector_u, detector_v


def diffracted_cone(u, alpha, gamma_span,
                    gamma_zero_direction=np.array((0, 0, 1)),
                    reshape=True):
    """ Returns diffracted directions by a powder (cone)


        Parameters
        ----------
        u : (N, dim) array
            directions of the incident rays
        alpha : float, radian
            half angle of the cone
        gamma_span : array, radian
            angle at which new rays are diffracted
        gamma_zero_direction : vector
            direction along which gamma=0
            toward gamma>0
        reshape : boolean
            if True return a (N * len(gamma_span), dim) array
            else a (N, len(gamma_span), dim) array

        Returns
        -------
        d : array (N * len(gamma_span), dim) or (N * len(gamma_span), dim)
            diffracted directions
    """

    gamma_span = np.asarray(gamma_span)

    # Define a new base
    u = normalize(u)
    w = normalize(np.cross(u, gamma_zero_direction))
    v = np.cross(w, u)

    cos_gamma = np.cos(gamma_span)
    sin_gamma = np.sin(gamma_span)
    rho = np.tan(alpha)

    # diffracted directions:
    d = u[:, :, np.newaxis] + \
        v[:, :, np.newaxis] * rho * cos_gamma + \
        w[:, :, np.newaxis] * rho * sin_gamma

    d = d.swapaxes(1, 2)  # coordinates along the last dimension

    if reshape:
        d = d.reshape(-1, 3, order='C')
    #  ‘C’ means to read / write the elements ... with the last axis index changing fastest,

    return d


# test
u_test = np.array(((1, 0, 0), (-10, 0, 0)))

d = diffracted_cone(u_test, np.pi / 4, (0, np.pi / 2, -np.pi / 2, np.pi / 3), reshape=True)
d.shape



def planar_powder_diffraction(A, u, deuxtheta, omega, gamma_span):
    """ Diffraction by a planar powder

    TODO: material and Bragg law here ?
    Parameters
    ----------
    A : (N, dim) array
        starting points of the incident rays
    u : (N, dim) array
        directions of the incident rays
    deuxtheta : float, radian
        diffracted angle
    omega : float, radian
        sample orientation
    gamma_span : (n, ) vector, radian
        out-of-plane diffracted angles

    """
    sample_position = normalize((0, 0, 0))
    sample_normal = normalize((+np.sin(omega), 0, +np.cos(omega)))

    # Intersection of incident beam with sample plane:
    B = plane_intersection(A, u, sample_position, sample_normal)

    d = diffracted_cone(u, deuxtheta,
                        gamma_span, reshape=True)

    # Broadcast B to the shape of d
    nbr_gamma_rays = len(gamma_span)
    B_prime = np.repeat(B[:, :, np.newaxis], nbr_gamma_rays, 2)

    B_prime = B_prime.swapaxes(1, 2)  # coordinates along the last dimension
    B_prime = B_prime.reshape(-1, 3, order='C')

    return B_prime, d


def source(nbr_rays, beam_shape, beam_divergence, rayon_gonio):
    Ax = np.ones((nbr_rays, 1)) * rayon_gonio
    Ay = (np.random.randn(nbr_rays, 1)) * beam_shape[1]  # Gaussian
    Az = (np.random.rand(nbr_rays, 1) - 0.5) * beam_shape[0]  # mm, Square

    u_incident_x = -np.ones((nbr_rays, 1))
    u_incident_y = np.random.randn(nbr_rays, 1) * beam_divergence[1]  # divergence
    u_incident_z = np.random.randn(nbr_rays, 1) * beam_divergence[0]

    A = np.hstack([Ax, Ay, Az])
    u = normalize(np.hstack([u_incident_x, u_incident_y, u_incident_z]))

    return A, u


