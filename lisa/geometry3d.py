import numpy as np


def translate(point, vector, length=None):
    vector = np.array(vector)
    if length is not None:
        vector = length * vector / np.linalg.norm(vector)
    return (np.array(point) + vector).tolist()


def circle(center, perp_vect, radius, element_number=10):
    """
    Function computed the circle points. No drawing.
    perp_vect is vector perpendicular to plane of circle
    """
    # tl = [0, 0.2, 0.4, 0.6, 0.8]
    tl = np.linspace(0, 1, element_number)

    # vector form center to edge of circle
    # u is a unit vector from the centre of the circle to any point on the
    # circumference

    # normalized perpendicular vector
    n = perp_vect / np.linalg.norm(perp_vect)

    # normalized vector from the centre to point on the circumference
    u = perpendicular_vector(n)
    u = u / np.linalg.norm(u)

    pts = []

    for t in tl:
        # u = np.array([0, 1, 0])
        # n = np.array([1, 0, 0])
        pt = radius * np.cos(t * 2 * np.pi) * u +\
            radius * np.sin(t * 2 * np.pi) * np.cross(u, n) +\
            center

        pt = pt.tolist()
        pts.append(pt)

    return pts


def perpendicular_vector(v):
    r""" Finds an arbitrary perpendicular vector to *v*."""
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])


def cylinder_circles(nodeA, nodeB, radius, element_number=10):
    """
    Return list of two circles with defined parameters.
    """

    vector = (np.array(nodeA) - np.array(nodeB)).tolist()
    ptsA = circle(nodeA, vector, radius, element_number)
    ptsB = circle(nodeB, vector, radius, element_number)

    return ptsA, ptsB
