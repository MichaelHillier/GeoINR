import os
import pandas as pd
import numpy as np
import torch
from geoinr.input.readers import reader_xml_polydata_file
from geoinr.utils.vtk_utils import add_np_property_to_vtk_object, create_vtk_polydata_from_coords_and_property
from vtkmodules.all import vtkPolyData


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


def convert_vtkPolyData_normals_to_dataframe(poly: vtkPolyData):
    """
    :param poly: vtkPolyData data structure containing geometric data (x, y, z) attributed with training label data
    :return: pandas dataframe [x, y, z, class_id]
    """

    normal_data = poly.GetPointData().GetArray(0)
    data = []
    for i in range(poly.GetNumberOfPoints()):
        x, y, z = poly.GetPoint(i)
        n = normal_data.GetTuple(i)
        nx = n[0]
        ny = n[1]
        nz = n[2]
        l = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx /= l
        ny /= l
        nz /= l
        row = [x, y, z, nx, ny, nz]
        data.append(row)

    normal_df = pd.DataFrame(data, columns=['x', 'y', 'z', 'nx', 'ny', 'nz'])

    return normal_df


class NormalData(object):
    """Dataset manager for Normal (bedding) data points"""

    def __init__(self, *args):
        self.normal_vtk = None
        self.normal_bounds = None
        self.normal_df = None
        self.n_normal_pts = 0
        self.coords = None
        self.normal_data = None

        self.normal_pred = None
        self.residuals = None

        self.build_data_from_vtk_file(args[0])

    def build_data_from_vtk_file(self, normal_file: str):
        if not os.path.isfile(normal_file):
            raise ValueError('File for interface data does not exist')
        self.normal_vtk = reader_xml_polydata_file(normal_file)
        # Get data bounds [xmin, xmax, ymin, ymax, zmin, zmax]
        self.normal_bounds = self.normal_vtk.GetBounds()
        # Create pandas dataframe
        self.normal_df = convert_vtkPolyData_normals_to_dataframe(self.normal_vtk)
        self.coords = self.normal_df[['x', 'y', 'z']].to_numpy()
        # Get normal data
        self.normal_data = self.normal_df[['nx', 'ny', 'nz']].to_numpy()
        self.n_normal_pts = self.normal_data.shape[0]

    def get_coords(self):
        return self.coords

    def transform(self, scalar):
        self.coords = scalar.transform(self.coords)
        nx = self.normal_data[:, 0]
        ny = self.normal_data[:, 1]
        nz = self.normal_data[:, 2]
        new_nx = nx * scalar.scale_[0]
        new_ny = ny * scalar.scale_[1]
        new_nz = nz * scalar.scale_[2]
        l = np.sqrt(new_nx * new_nx + new_ny * new_ny + new_nz * new_nz)
        new_nx /= l
        new_ny /= l
        new_nz /= l
        normals_transform = np.stack((new_nx, new_ny, new_nz), axis=1)
        self.normal_data = normals_transform

    def convert_to_torch(self):
        self.coords = torch.from_numpy(self.coords).float()
        self.normal_data = torch.from_numpy(self.normal_data).float()

    def send_to_gpu(self, rank):
        self.coords = self.coords.to(rank)
        self.normal_data = self.normal_data.to(rank)

    def set_normal_pred(self, normal_pred):
        self.normal_pred = normal_pred

    def set_residuals(self, residuals):
        self.residuals = residuals

    def __add_properties_to_vtk_object_if_present(self):
        added_properties = False
        if self.normal_pred is not None:
            added_properties = True
            add_np_property_to_vtk_object(self.normal_vtk, "normal_pred", self.normal_pred)
        if self.residuals is not None:
            added_properties = True
            add_np_property_to_vtk_object(self.normal_vtk, "residuals", self.residuals)
        if added_properties:
            return self.normal_vtk
        else:
            return None

    def process_model_outputs(self):
        self.normal_vtk = self.__add_properties_to_vtk_object_if_present()