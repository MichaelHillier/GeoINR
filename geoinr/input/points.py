import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_nearest_neighbor_dist_from_pts(coords: np.ndarray):
    neighbors = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(coords)
    neigh_dist, indices = neighbors.kneighbors(coords)
    neigh_dist = neigh_dist[:, 1]
    return neigh_dist


def get_bounds_from_coords(coords: np.ndarray, xy_buffer=None, z_buffer=None):
    assert coords.ndim == 2, "input coords array is not 2D array"
    assert coords.shape[1] == 3, "input coords are not 3D"

    coord_min = coords.min(axis=0)
    coord_max = coords.max(axis=0)

    x_min = coord_min[0]
    x_max = coord_max[0]
    y_min = coord_min[1]
    y_max = coord_max[1]
    z_min = coord_min[2]
    z_max = coord_max[2]
    bounds = np.array([x_min, x_max, y_min, y_max, z_min, z_max])
    if xy_buffer != 0 or z_buffer != 0:
        if xy_buffer == 0:
            xy_buffer = z_buffer
        if z_buffer == 0:
            z_buffer = xy_buffer
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        dz = bounds[5] - bounds[4]
        bounds[0] = bounds[0] - xy_buffer * dx
        bounds[1] = bounds[1] + xy_buffer * dx
        bounds[2] = bounds[2] - xy_buffer * dy
        bounds[3] = bounds[3] + xy_buffer * dy
        bounds[4] = bounds[4] - z_buffer * dz
        bounds[5] = bounds[5] + z_buffer * dz

    return bounds


def concat_coords_from_datasets(*datasets):
    coords_list = []
    for dataset_i in datasets:
        assert type(dataset_i) == np.ndarray, "coord dataset is not a ndarray"
        assert dataset_i.ndim == 2, "input dataset is not 2D"
        coords_list.append(dataset_i)

    all_coords = np.vstack(coords_list)
    return all_coords