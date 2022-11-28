import torch
from torch.utils.data import Dataset
import numpy as np
from vtkmodules.all import vtkStructuredGrid, vtkImageData, vtkPoints, vtkCellCenters
from vtkmodules.util import numpy_support
from geoinr.utils.vtk_utils import add_np_property_to_vtk_object


def generate_vtk_structure_grid_and_grid_points(bounds, xy_resolution, z_resolution):
    if bounds is not None:
        if isinstance(bounds, np.ndarray):
            if bounds.size != 6:
                raise ValueError('Bound array is not the appropriate size')
            else:
                xmin = bounds[0]
                xmax = bounds[1]
                ymin = bounds[2]
                ymax = bounds[3]
                zmin = bounds[4]
                zmax = bounds[5]

    nx = int((xmax - xmin) / xy_resolution) + 1
    ny = int((ymax - ymin) / xy_resolution) + 1
    nz = int((zmax - zmin) / z_resolution) + 1
    dims = np.array([nx, ny, nz])
    sample_x = np.linspace(xmin, xmax, nx)
    sample_y = np.linspace(ymin, ymax, ny)
    sample_z = np.linspace(zmin, zmax, nz)
    zz, yy, xx = np.meshgrid(sample_z, sample_y, sample_x, indexing='ij')
    grid_pts = np.vstack((xx, yy, zz)).reshape(3, -1).T

    # Create VTK Structured Grid
    sgrid = vtkStructuredGrid()
    sgrid.SetDimensions(dims[0], dims[1], dims[2])
    points = vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(grid_pts, deep=True))
    sgrid.SetPoints(points)

    return grid_pts, sgrid


def generate_vtk_imagedata_grid_and_grid_points(bounds, xy_resolution, z_resolution):
    if isinstance(bounds, np.ndarray):
        if bounds.size != 6:
            raise ValueError('Bound array is not the appropriate size')
        else:
            xmin = bounds[0]
            xmax = bounds[1]
            ymin = bounds[2]
            ymax = bounds[3]
            zmin = bounds[4]
            zmax = bounds[5]

    nx = int((xmax - xmin) / xy_resolution) + 1
    ny = int((ymax - ymin) / xy_resolution) + 1
    nz = int((zmax - zmin) / z_resolution) + 1
    origin = np.array([xmin, ymin, zmin])

    grd = vtkImageData()
    grd.SetOrigin(origin)
    grd.SetSpacing(xy_resolution, xy_resolution, z_resolution)
    grd.SetDimensions(nx, ny, nz)

    centers = vtkCellCenters()
    centers.SetInputData(grd)
    centers.Update()
    centers = numpy_support.vtk_to_numpy(centers.GetOutput().GetPoints().GetData())

    return centers, grd


class Grid(object):
    """Dataset manager for Grid points. Actual grid coordinates are computed prior and used as input to create this
     object. Grid should be rectilinear. Points are ordered in the following manner: x-axis fastest, y-axis next, z-axis
     slowest.
     Coordinate Scaling/Normalization occurs after object created
     Dataset includes:
     coords: a matrix containing all 3D grid points [x, y, z]"""

    def __init__(self, grid_coords, grid=None, series=None):
        self.coords = grid_coords
        self.n_grid_pts = self.coords.shape[0]
        self.series = series
        self.scalar_pred = None
        self.scalar_series = None
        self.scalar_grad_pred = None
        self.scalar_grad_norm_pred = None
        self.unit_pred = None
        self.grid_vtk = grid
        if grid is not None:
            self.dims = grid.GetDimensions()
        else:
            self.dims = None
        self.sampled_grid_points = None

    def __len__(self):
        return self.n_grid_pts

    def __getitem__(self, idx):
        return self.coords[idx]

    def get_coords(self):
        return self.coords

    def transform(self, scalar):
        self.coords = scalar.transform(self.coords)

    def transform_sampled_grid_points(self, scalar):
        assert self.sampled_grid_points is not None, "there are no sampled grid points"
        self.sampled_grid_points = scalar.transform(self.sampled_grid_points)

    def send_to_gpu(self, rank):
        self.coords = self.coords.to(rank)

    def set_scalar_pred(self, scalar_pred):
        self.scalar_pred = scalar_pred

    def set_scalar_series(self, scalar_series):
        self.scalar_series = scalar_series

    def set_scalar_grad_pred(self, scalar_grad_pred):
        self.scalar_grad_pred = scalar_grad_pred

    def set_scalar_grad_norm_pred(self, scalar_grad_norm_pred):
        self.scalar_grad_norm_pred = scalar_grad_norm_pred

    def set_unit_pred(self, unit_pred):
        self.unit_pred = unit_pred

    def uniformly_sample_grid_for_points(self, n_points):
        assert self.dims is not None, "grid dimensions are not set"
        # proportion of points along each dimension compared to the max number of points along a particular direction
        max_n_pts_dir = max(self.dims)
        px = self.dims[0] / max_n_pts_dir
        py = self.dims[1] / max_n_pts_dir
        pz = self.dims[2] / max_n_pts_dir
        # to uniformly sample the grid to get a total of n_points WHILE also respecting the ratio of points along two
        # directions the equation is
        # (x * px) ( x * py ) * ( x * pz ) = n_points, solve for x
        x = (((max_n_pts_dir ** 3) * n_points) / (self.dims[0] * self.dims[1] * self.dims[2])) ** (1.0 / 3.0)
        nx = round(x * px)
        ny = round(x * py)
        nz = round(x * pz)
        bounds = self.grid_vtk.GetBounds()
        sample_x = np.linspace(bounds[0], bounds[1], nx)
        sample_y = np.linspace(bounds[2], bounds[3], ny)
        sample_z = np.linspace(bounds[4], bounds[5], nz)
        zz, yy, xx = np.meshgrid(sample_z, sample_y, sample_x, indexing='ij')
        self.sampled_grid_points = np.vstack((xx, yy, zz)).reshape(3, -1).T

    def __add_properties_to_vtk_object_if_present(self):
        assert self.grid_vtk is not None, "there is no grid vtk object"
        if self.scalar_pred is not None:
            add_np_property_to_vtk_object(self.grid_vtk, "Scalar Field", self.scalar_pred)
        if self.scalar_series is not None:
            n_series = self.scalar_series.shape[1]
            for i in range(n_series):
                series_i = self.scalar_series[:, i]
                series_name = "Scalar Field" + str(i)
                add_np_property_to_vtk_object(self.grid_vtk, series_name, series_i)
        if self.scalar_grad_pred is not None:
            add_np_property_to_vtk_object(self.grid_vtk, "Scalar Gradient", self.scalar_grad_pred)
        if self.scalar_grad_norm_pred is not None:
            add_np_property_to_vtk_object(self.grid_vtk, "Scalar Gradient Norm", self.scalar_grad_norm_pred)
        if self.unit_pred is not None:
            add_np_property_to_vtk_object(self.grid_vtk, "Unit", self.unit_pred, continuous=False)
        return self.grid_vtk

    def process_model_outputs(self, map_to_original_class_ids=None):
        # 1) remap unit predictions to original class ids
        if isinstance(self.unit_pred, np.ndarray):
            if isinstance(map_to_original_class_ids, dict):
                self.unit_pred = np.vectorize(map_to_original_class_ids.get)(self.unit_pred)
        # 2) add model properties to vtk object
        self.grid_vtk = self.__add_properties_to_vtk_object_if_present()


class GridPointDataDistributedSampler(object):
    def __init__(self, grid_dataset: Grid, ngpus):
        """ Evenly splits grids points into n = ngpus pieces. Each piece will be sent to different gpus for parallelized
        inference/prediction on the model after training.
        Assumed the coords/features are normalized/scaled already"""
        self.ngpus = ngpus
        indices = np.arange(grid_dataset.n_grid_pts)
        split_indices = np.array_split(indices, ngpus)
        self.subset = {}
        for i in range(ngpus):
            grid_coord_subset = grid_dataset.coords[split_indices[i]]
            self.subset[i] = Grid(torch.from_numpy(grid_coord_subset).float())

    def get_subset(self, i):
        return self.subset[i]


class GridData(Dataset):
    def __init__(self, coords):
        self.coords = coords
        self.n_grid_pts = self.coords.shape[0]

    def __len__(self):
        return self.n_grid_pts

    def __getitem__(self, idx):
        return self.coords[idx]
