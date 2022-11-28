import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from vtkmodules.all import vtkPolyData, vtkXMLPolyDataReader, vtkBoundingBox
from geoinr.input.points import compute_nearest_neighbor_dist_from_pts
from geoinr.input.grids import generate_vtk_structure_grid_and_grid_points

# These are deprecated methods. Will be removed at some point.


def reader_xml_polydata_file(pd_filename: str):
    """
    :param pd_filename: an XML vtkPolyData file format (*.vtp)
    :return: vtkPolyData data structure - also carries any property data attributed to points, cells, as well as field
             data (meta data)
    """
    if not os.path.isfile(pd_filename):
        raise ValueError('File does not exist')

    reader = vtkXMLPolyDataReader()
    reader.SetFileName(pd_filename)
    reader.Update()

    return reader.GetOutput()


def convert_vtkPolydata_normal_to_dataframe(poly: vtkPolyData):
    """
    :param poly: vtkPolyData data structure containing geometric data (x, y, z) attributed with normal data
    :return: pandas dataframe [x, y, z, nx, ny, nz]
    """
    normal_data = poly.GetPointData().GetArray(0)
    data = []
    for i in range(poly.GetNumberOfPoints()):
        x, y, z = poly.GetPoint(i)
        nx, ny, nz = normal_data.GetTuple(i)
        row = [x, y, z, nx, ny, nz]
        data.append(row)

    normal_df = pd.DataFrame(data, columns=['x', 'y', 'z', 'nx', 'ny', 'nz'])

    return normal_df


def convert_vtkPolydata_contact_to_dataframe(poly: vtkPolyData):
    """
    :param poly: vtkPolyData data structure containing geometric data (x, y, z) attributed with level data
    :return: pandas dataframe [x, y, z, level]
    """
    level_data = poly.GetPointData().GetArray(0)
    data = []
    for i in range(poly.GetNumberOfPoints()):
        x, y, z = poly.GetPoint(i)
        level = level_data.GetTuple1(i)
        row = [x, y, z, level]
        data.append(row)

    contact_df = pd.DataFrame(data, columns=['x', 'y', 'z', 'level'])

    return contact_df


def convert_vtkPolyData_labels_to_dataframe(poly: vtkPolyData):
    """
    :param poly: vtkPolyData data structure containing geometric data (x, y, z) attributed with training label data
    :return: pandas dataframe [x, y, z, class_id]
    """

    unit_data = poly.GetPointData().GetArray(0)
    data = []
    for i in range(poly.GetNumberOfPoints()):
        x, y, z = poly.GetPoint(i)
        unit = unit_data.GetTuple1(i)
        row = [x, y, z, unit]
        data.append(row)

    units_df = pd.DataFrame(data, columns=['x', 'y', 'z', 'unit'])

    return units_df


def get_constraint_dataframes(interface_file, normal_file, rock_units_file, check_collocations=False):
    """
    :interface_file: vtk file containing geological interface data. Assumes the first data array holds interface info
    e.g. scalar_field, level, strat ... a scalar property (1D)
    :normal_file: vtk file containing planar orientation data. Assumes the first data array holds the normal data
    e.g. Normal, normal, planar... a vector property (3D)
    : rock_units_files: a vtk file containing geological class data. Assumes the first data array holds the class info
    (an integer). e.g. class, zone, unit, geological_units ... an integer property (1D)
    : return: 3 pandas dataframe containing interface, normal, and class data. In addition the bounds [xmin, xmax, ymin,
    ymax, zmin, zmax] of all the data is returned. Note if any of the 3 types of data files is None, then the returned
    pandas dataframe is also None.
    """

    interface_df = None
    normal_df = None
    rock_units_df = None
    bounding_box = vtkBoundingBox()
    if interface_file is not None:
        if not os.path.isfile(interface_file):
            raise ValueError('File for interface data does not exist')
        interface_vtk = reader_xml_polydata_file(interface_file)
        interface_bounds = interface_vtk.GetBounds()
        bounding_box.AddBounds(interface_bounds)
        interface_df = convert_vtkPolydata_contact_to_dataframe(interface_vtk)
        if check_collocations:
            # check for collocation, and remove if necessary
            interface_position = interface_df[['x', 'y', 'z']].to_numpy()
            neigh_dist = compute_nearest_neighbor_dist_from_pts(interface_position)
            nn_dist_min = neigh_dist.min()
            if nn_dist_min < 0.1:
                collocated_indices = np.where(neigh_dist < 0.1)[0]
                # remove these points
                interface_df = interface_df.drop(collocated_indices)

    if normal_file is not None:
        if not os.path.isfile(normal_file):
            raise ValueError('File for normal data does not exist')
        normal_vtk = reader_xml_polydata_file(normal_file)
        normal_bounds = normal_vtk.GetBounds()
        bounding_box.AddBounds(normal_bounds)
        normal_df = convert_vtkPolydata_normal_to_dataframe(normal_vtk)
        if check_collocations:
            # check for collocation, and remove if necessary
            normal_position = normal_df[['x', 'y', 'z']].to_numpy()
            neigh_dist = compute_nearest_neighbor_dist_from_pts(normal_position)
            nn_dist_min = neigh_dist.min()
            if nn_dist_min < 0.1:
                collocated_indices = np.where(neigh_dist < 0.1)[0]
                # remove these points
                normal_df = normal_df.drop(collocated_indices)

    if rock_units_file is not None:
        if not os.path.isfile(rock_units_file):
            raise ValueError('File for rock units data does not exist')
        rock_units_vtk = reader_xml_polydata_file(rock_units_file)
        rock_units_bounds = rock_units_vtk.GetBounds()
        bounding_box.AddBounds(rock_units_bounds)
        rock_units_df = convert_vtkPolyData_labels_to_dataframe(rock_units_vtk)
        if check_collocations:
            # check for collocation, and remove if necessary
            rock_position = rock_units_df[['x', 'y', 'z']].to_numpy()
            neigh_dist = compute_nearest_neighbor_dist_from_pts(rock_position)
            nn_dist_min = neigh_dist.min()
            if nn_dist_min < 0.1:
                collocated_indices = np.where(neigh_dist < 0.1)[0]
                # remove these points
                rock_units_df = rock_units_df.drop(collocated_indices)

    if all(constraint_type is None for constraint_type in [interface_file, normal_file, rock_units_file]):
        raise IOError('There are no constraints supplied, or not loaded properly')

    bounds_min = bounding_box.GetMinPoint()
    bounds_max = bounding_box.GetMaxPoint()
    bounds = [bounds_min[0],
              bounds_max[0],
              bounds_min[1],
              bounds_max[1],
              bounds_min[2],
              bounds_max[2]]

    return interface_df, normal_df, rock_units_df, bounds


class PointData(object):
    r""" Python object containing various attributes of input geological data and grid points

    """

    def __init__(self, features, interface_data, interface_indices,
                 normal_data, normal_indices,
                 grid_points, grid_indices, grid,
                 horizon_interface_indices, scalar):
        self.features = features
        self.interface_data = interface_data
        self.interface_indices = interface_indices
        self.normal_data = normal_data
        self.normal_indices = normal_indices
        self.grid_points = grid_points
        self.grid_indices = grid_indices
        self.grid = grid
        self.horizon_interface_indices = horizon_interface_indices
        self.scalar = scalar
        self.have_interface = True
        self.have_orientation = True

        self.convert_to_pytorch()

    def convert_to_pytorch(self):

        self.features = torch.from_numpy(self.features).float()
        if self.interface_data is not None:
            self.interface_data = torch.from_numpy(self.interface_data).float()
            self.interface_indices = torch.LongTensor(self.interface_indices)
        else:
            self.have_interface = False
        if self.normal_data is not None:
            self.normal_data = torch.from_numpy(self.normal_data).float()
            self.normal_indices = torch.LongTensor(self.normal_indices)
        else:
            self.have_orientation = False
        self.grid_indices = torch.LongTensor(self.grid_indices)

    def transform_position_error(self, dx, dy, dz):
        pt_0 = np.array([0.0, 0.0, 0.0]).reshape(1, -1)
        pt_x = np.array([dx, 0.0, 0.0]).reshape(1, -1)
        pt_y = np.array([0.0, dy, 0.0]).reshape(1, -1)
        pt_z = np.array([0.0, 0.0, dz]).reshape(1, -1)
        pt_0_t = self.scalar.transform(pt_0)
        pt_x_t = self.scalar.transform(pt_x)
        pt_y_t = self.scalar.transform(pt_y)
        pt_z_t = self.scalar.transform(pt_z)
        dx_t = (pt_x_t - pt_0_t)[0][0]
        dy_t = (pt_y_t - pt_0_t)[0][1]
        dz_t = (pt_z_t - pt_0_t)[0][2]

        return dx_t, dy_t, dz_t

    def add_positional_noise_to_features(self, dx, dy, dz):
        dx, dy, dz = self.transform_position_error(dx, dy, dz)
        device = self.features.device
        std = torch.tensor([dx, dy, dz]).type(torch.FloatTensor).to(device)
        noisy = torch.normal(mean=self.features, std=std)
        return noisy

    def send_to_gpu(self, device):
        self.features = self.features.to(device)
        if self.have_interface:
            self.interface_data = self.interface_data.to(device)
            self.interface_indices = self.interface_indices.to(device)
        if self.have_orientation:
            self.normal_data = self.normal_data.to(device)
            self.normal_indices = self.normal_indices.to(device)
        self.grid_indices = self.grid_indices.to(device)


def load_input_data_and_grid(args):
    '''
    :param args: command line arguments for additional processing
    :return: dataset containing constraints and domain
    '''

    interface_file = os.path.join(args.root_dir, args.datadir, args.dataset, 'interface.vtp')
    normal_file = os.path.join(args.root_dir, args.datadir, args.dataset, 'normal.vtp')
    class_file = None

    if not os.path.isfile(interface_file):
        interface_file = None
    if not os.path.isfile(normal_file):
        normal_file = None

    # Create interface, normal, and unit dataframes (pandas)
    interface_df, normal_df, units_df, data_bounds = get_constraint_dataframes(interface_file, normal_file,
                                                                               class_file, True)

    # Combine All dataframes
    dfs_list = [interface_df, normal_df, units_df]
    Not_none_dfs = filter(None.__ne__, dfs_list)
    dfs_list = list(Not_none_dfs)
    constraints = pd.concat(dfs_list, ignore_index=True)

    # Class Data
    have_classes = False
    if 'class_id' in constraints.columns:
        have_classes = True
        class_values = constraints['class_id']
        class_values = class_values[class_values.notna()].unique()  # Grab unique class values (with nan removed)
        class_values.sort()  # Sort unique class values lowest->highest
        num_classes = class_values.size  # Number of unique classes
        class_ids = np.arange(num_classes)  # [0, ..., num_classes - 1]
        classes = dict(zip(class_values, class_ids))  #

    # Get interface data
    interface_pos = None
    interface_data = None
    n_interface = 0
    interface_indices = None
    if interface_df is not None:
        interface_pos = interface_df[['x', 'y', 'z']].to_numpy()
        interface_data = interface_df[['level']].to_numpy()
        scaler = MinMaxScaler((-1, 1))
        scaler.fit(interface_data)
        interface_data = scaler.transform(interface_data)
        interface_data = interface_data.flatten()
        n_interface = interface_data.shape[0]
        unique_interface_values = np.unique(interface_data)
        # resort unique values in descending order
        unique_interface_values = unique_interface_values[::-1]
        interface_indices = [i for i in range(n_interface)]
        horizon_interface_indices = [[i for i in range(n_interface) if interface_data[i] == value]
                                     for value in unique_interface_values]

    # Get normal data
    normal_pos = None
    normal_data = None
    n_normal = 0
    normal_indices = None
    if normal_df is not None:
        normal_pos = normal_df[['x', 'y', 'z']].to_numpy()
        normal_data = normal_df[['nx', 'ny', 'nz']].to_numpy()
        n_normal = normal_data.shape[0]
        normal_indices = [i + n_interface for i in range(n_normal)]

    # Generate Grid Pts
    dx = data_bounds[1] - data_bounds[0]
    dy = data_bounds[3] - data_bounds[2]
    dz = data_bounds[5] - data_bounds[4]
    bounds = np.zeros(6)
    bounds[0] = data_bounds[0] - args.xy_buffer * dx
    bounds[1] = data_bounds[1] + args.xy_buffer * dx
    bounds[2] = data_bounds[2] - args.xy_buffer * dy
    bounds[3] = data_bounds[3] + args.xy_buffer * dy
    bounds[4] = data_bounds[4] - args.xy_buffer * dz
    bounds[5] = data_bounds[5] + args.xy_buffer * dz
    grid_pts, grid = generate_vtk_structure_grid_and_grid_points(bounds, args.xy_resolution, args.z_resolution)

    n_grid_pts = grid_pts.shape[0]
    grid_indices = [(i + (n_interface + n_normal)) for i in range(n_grid_pts)]

    pos_tuple = (interface_pos, normal_pos, grid_pts)
    Not_none_tuples = filter(None.__ne__, pos_tuple)
    pos_tuple = tuple(Not_none_tuples)
    feats = np.vstack(pos_tuple)

    if args.scale_isometric:
        coord_min = feats.min(axis=0)
        coord_max = feats.max(axis=0)

        coord_range = coord_max - coord_min
        range_max = coord_range.max()

        # center data, then scale (divide each translated coordinate by the longest axis/2 - divide by 2 so that the longest
        # axis range is between -1 and 1 : 1 - (-1) = 2
        feats = (feats - (coord_min + coord_range / 2)) / (range_max / 2)
        coord_scale_factor = 1.0 / (range_max / 2.0)
        scalar = None
    else:
        constraints_position = constraints[['x', 'y', 'z']].to_numpy()
        scalar = StandardScaler()
        #scalar = MinMaxScaler((-1, 1))
        scalar.fit(constraints_position)
        feats = scalar.transform(feats)

    return PointData(feats, interface_data, interface_indices, normal_data, normal_indices,
                     grid_pts, grid_indices, grid, horizon_interface_indices, scalar)