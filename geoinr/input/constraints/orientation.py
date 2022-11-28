import numpy as np


def orientation_pca(normals):
    """
    :param normals: Matrix containing normals [N, 3]
                    Each row is a normal vector [nx, ny, nz]
    :return: Sorted eigenvalues and eigenvalues representing principal directions of anisotropy
    """
    # S is the dispersion/orientation matrix for the orientation dataset
    S = np.zeros((3, 3))
    N = np.shape(normals)[0]
    mean_dir = np.zeros(3)
    for normal in normals:
        mean_dir += normal
        S += np.matmul(normal.reshape(3, 1), normal.reshape(1, 3))
    S /= N
    mean_dir /= N
    e_val, e_vec = np.linalg.eig(S)
    # sort the eigen system from highest to lowest
    # e.g. eval1 > eval2 > eval3 | eval1 <=> evec1, eval2 <=> evec2, eval3 <=> evec3
    # <=> 'associated'
    # evec1 : mean axes (cross plunge - maximal strain direction)
    # evec2 : major axes (fold hinge - layer displayment direction)
    # evec3 : minor axes (plunge direction - direction which structure varies the least)
    idx = e_val.argsort()[::-1]
    e_val = e_val[idx]
    e_vec = e_vec[:, idx]
    mean_dir_norm = np.linalg.norm(mean_dir)

    return e_val, e_vec