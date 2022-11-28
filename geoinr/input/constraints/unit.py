import os
import pandas as pd
import numpy as np
import torch
from geoinr.input.readers import reader_xml_polydata_file
from geoinr.utils.vtk_utils import add_np_property_to_vtk_object, create_vtk_polydata_from_coords_and_property
from vtkmodules.all import vtkPolyData
from sklearn.model_selection import KFold


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


class UnitData(object):
    """Dataset manager for Lithological/Stratrigraphic (discrete) points"""

    def __init__(self, *args):
        self.unit_vtk = None
        self.unit_bounds = None
        self.unit_df = None
        self.n_class_pts = 0
        self.n_classes = 0
        self.coords = None
        self.unit_data = None
        self.unit_dict = None
        self.unique_unit_ids = None
        self.class_weights = None
        self.unit_id_to_unit_level = {}
        self.unit_indices = None
        self.original_unit_indices = None

        self.unit_pred = None
        self.residuals = None
        self.class_residual_means = None
        self.unit_metric_dict = None

        if len(args) == 1:
            coords, unit_data = self.build_data_from_vtk_file(args[0])
            self.build_from_coords_and_unit_data(coords, unit_data)
        elif len(args) == 2:
            if isinstance(args[0], str):
                coords, unit_data = self.build_data_from_vtk_file(args[0])
                self.build_from_coords_unit_data_and_unit_dict(coords, unit_data, args[1])
            elif isinstance(args[0], np.ndarray):
                self.build_from_coords_and_unit_data(args[0], args[1])
        elif len(args) == 3:
            self.build_from_coords_unit_data_and_unit_mapping(args[0], args[1], args[2])
        else:
            raise IOError("UnitData: Unexpected number of input arguments, supposed to be 1")

    def build_data_from_vtk_file(self, unit_file: str):
        if not os.path.isfile(unit_file):
            raise ValueError('File for interface data does not exist')
        self.unit_vtk = reader_xml_polydata_file(unit_file)
        # Get data bounds [xmin, xmax, ymin, ymax, zmin, zmax]
        self.unit_bounds = self.unit_vtk.GetBounds()
        # Create pandas dataframe
        self.unit_df = convert_vtkPolyData_labels_to_dataframe(self.unit_vtk)
        coords = self.unit_df[['x', 'y', 'z']].to_numpy()
        # Get interface data (stratigraphically sequenced scalar values s. Larger s means younger, smaller s means older
        unit_data = self.unit_df[['unit']].to_numpy()
        return coords, unit_data

    def build_from_coords_and_unit_data(self, coords: np.ndarray, unit_data: np.ndarray):
        self.coords = coords
        self.n_class_pts = unit_data.shape[0]
        unique_unit_ids, counts = np.unique(unit_data, return_counts=True)
        self.n_classes = unique_unit_ids.shape[0]
        max_count = counts.max()
        min_count = counts.min()
        self.class_weights = max_count / counts
        # Check if unique unit id (integer) are completely sequential. e.g. there is are no missing integers
        min_unit_id = unique_unit_ids.min()
        max_unit_id = unique_unit_ids.max()
        sequential_units = np.arange(min_unit_id, max_unit_id + 1, 1)
        if not np.array_equal(sequential_units, unique_unit_ids):
            # find the classes there are NOT sampled by the data
            missing_classes = np.setdiff1d(sequential_units, unique_unit_ids)
            # create sorted dictionary for the existing class weights
            class_weight_dict = {}
            for i in range(unique_unit_ids.size):
                class_weight_dict[unique_unit_ids[i]] = self.class_weights[i]
            # insert missing classes into dict
            for missing_class in missing_classes:
                class_weight_dict[missing_class] = 1.0
            class_weight_dict = dict(sorted(class_weight_dict.items()))
            self.class_weights = np.fromiter(class_weight_dict.values(), dtype=float)
            unique_unit_ids = np.fromiter(class_weight_dict.keys(), dtype=float)
            self.n_classes = unique_unit_ids.size

        # Re-map unit ids to [0, ..., n_classes - 1] range
        unit_level_to_unit_id = {}
        self.unit_id_to_unit_level = {}
        for i, unit_id in enumerate(unique_unit_ids):
            unit_level_to_unit_id[unit_id] = i
            self.unit_id_to_unit_level[i] = unit_id
        self.unit_data = np.vectorize(unit_level_to_unit_id.get)(unit_data).flatten()
        self.unique_unit_ids = np.fromiter(self.unit_id_to_unit_level.keys(), dtype=float)

        self.unit_indices = {i: [] for i in range(self.n_classes)}
        for i in range(self.n_class_pts):
            self.unit_indices[self.unit_data[i]].append(i)

    def build_from_coords_unit_data_and_unit_dict(self, coords: np.ndarray, unit_data: np.ndarray,
                                                  unit_dict: dict):
        self.unit_dict = unit_dict
        self.coords = coords
        self.n_class_pts = unit_data.shape[0]
        unique_unit_ids, counts = np.unique(unit_data, return_counts=True)
        max_count = counts.max()
        min_count = counts.min()
        class_weight_dict = {unique_unit_ids[i]: (max_count / counts[i]) for i in range(unique_unit_ids.size)}
        # Check if unique unit id (integer) are consistent with unit_level (dict)
        assert set(unique_unit_ids).issubset(unit_dict.values()), "unexpected new value found in supplied unit dataset"
        # find the classes there are NOT sampled by the data
        missing_classes = np.setdiff1d(np.fromiter(unit_dict.values(), dtype=int), unique_unit_ids)
        for missing_class in missing_classes:
            class_weight_dict[missing_class] = 1.0
        class_weight_dict = dict(sorted(class_weight_dict.items()))
        self.class_weights = np.fromiter(class_weight_dict.values(), dtype=float)
        self.n_classes = len(unit_dict.values())
        # Re-map unit ids to [0, ..., n_classes - 1] range
        unit_level_to_unit_id = {}
        self.unit_id_to_unit_level = {}
        for unit_id, unit_level in unit_dict.items():
            unit_level_to_unit_id[unit_level] = unit_id
            self.unit_id_to_unit_level[unit_id] = unit_level
        self.unit_data = np.vectorize(unit_level_to_unit_id.get)(unit_data).flatten()
        self.unique_unit_ids = np.fromiter(self.unit_id_to_unit_level.keys(), dtype=float)

        self.unit_indices = {i: [] for i in range(self.n_classes)}
        for i in range(self.n_class_pts):
            self.unit_indices[self.unit_data[i]].append(i)

    def build_from_coords_unit_data_and_unit_mapping(self, coords: np.ndarray, unit_data: np.ndarray,
                                                     unit_mapping: dict):
        self.unit_id_to_unit_level = unit_mapping
        self.coords = coords
        self.n_class_pts = unit_data.shape[0]
        unique_unit_ids, counts = np.unique(unit_data, return_counts=True)
        self.unit_data = unit_data
        max_count = counts.max()
        min_count = counts.min()
        class_weight_dict = {unique_unit_ids[i]: (max_count / counts[i]) for i in range(unique_unit_ids.size)}
        # Check if unique unit id (integer) are consistent with unit_level (dict)
        assert set(unique_unit_ids).issubset(unit_mapping.keys()), "unexpected new value found in supplied unit dataset"
        # find the classes there are NOT sampled by the data
        missing_classes = np.setdiff1d(np.fromiter(unit_mapping.keys(), dtype=int), unique_unit_ids)
        for missing_class in missing_classes:
            class_weight_dict[missing_class] = 1.0
        class_weight_dict = dict(sorted(class_weight_dict.items()))
        self.class_weights = np.fromiter(class_weight_dict.values(), dtype=float)
        self.n_classes = len(unit_mapping.keys())
        self.unique_unit_ids = unique_unit_ids

        self.unit_indices = {i: [] for i in range(self.n_classes)}
        for i in range(self.n_class_pts):
            self.unit_indices[self.unit_data[i]].append(i)

    def __len__(self):
        return self.n_class_pts

    def __getitem__(self, idx):
        return self.coords[idx]

    def get_coords(self):
        return self.coords

    def transform(self, scalar):
        self.coords = scalar.transform(self.coords)

    def convert_to_torch(self):
        self.coords = torch.from_numpy(self.coords).float()
        self.unit_data = torch.from_numpy(self.unit_data).float()
        self.class_weights = torch.from_numpy(self.class_weights).float()

    def send_to_gpu(self, rank):
        self.coords = self.coords.to(rank)
        self.unit_data = self.unit_data.to(rank)
        self.class_weights = self.class_weights.to(rank)

    def set_unit_pred(self, unit_pred):
        self.unit_pred = unit_pred

    def set_residuals(self, residuals):
        self.residuals = residuals

    def set_class_residuals_means(self, class_residual_means):
        self.class_residual_means = class_residual_means

    def __compute_missing_units_from_predictions(self):
        if self.unit_pred is not None:
            unit_pred_units = np.unique(self.unit_pred)
            unique_units = np.fromiter(self.unit_id_to_unit_level.values(), dtype=float)
            overlap = np.isin(unique_units, unit_pred_units)
            indices_of_units_not_present = np.nonzero(overlap == False)[0]
            n_missing_units = indices_of_units_not_present.size
            if n_missing_units == 0:
                return None
            else:
                units_not_present = unique_units[indices_of_units_not_present]
                print(f'There are {n_missing_units} units missing in the model. The list of missing units are '
                      f': {units_not_present.tolist()}')
                return units_not_present

        else:
            return None

    def __generate_unit_metric_dict(self):
        missing_units = self.__compute_missing_units_from_predictions()
        unit_metric_dict = {}
        if missing_units is not None:
            unit_metric_dict['missing_units'] = missing_units
        if self.class_residual_means is not None:
            if isinstance(self.class_residual_means, dict):
                class_residual_mean_df = pd.DataFrame({"class_id": np.fromiter(self.unit_id_to_unit_level.values(),
                                                                               dtype=float),
                                                       "residual_mean": np.fromiter(self.class_residual_means.values(),
                                                                                    dtype=float)})
            else:
                class_residual_mean_df = pd.DataFrame({"class_id": np.fromiter(self.unit_id_to_unit_level.values(),
                                                                               dtype=float),
                                                       "residual_mean": self.class_residual_means})
            unit_metric_dict['class_residual_means'] = class_residual_mean_df
        if unit_metric_dict:
            return unit_metric_dict
        else:
            return None

    def __add_properties_to_vtk_object_if_present(self):
        added_properties = False
        if self.unit_pred is not None:
            added_properties = True
            add_np_property_to_vtk_object(self.unit_vtk, "unit_pred", self.unit_pred)
        if self.residuals is not None:
            added_properties = True
            add_np_property_to_vtk_object(self.unit_vtk, "residuals", self.residuals)
        if added_properties:
            return self.unit_vtk
        else:
            return None

    def process_model_outputs(self):
        # 1) remap unit predictions to original class ids
        if isinstance(self.unit_pred, np.ndarray):
            self.unit_pred = np.vectorize(self.unit_id_to_unit_level.get)(self.unit_pred)
        # 2) generate unit metric dictionary
        self.unit_metric_dict = self.__generate_unit_metric_dict()
        # 3) add model properties to vtk object
        self.unit_vtk = self.__add_properties_to_vtk_object_if_present()


class UnitKFoldSplit(object):
    def __init__(self, unit: UnitData, k, shuffle=False):
        self.split = []  # each element will be an InterfaceData object
        kf = KFold(n_splits=k, shuffle=shuffle)
        for train_index, test_index in kf.split(unit.coords):
            # have to re-index horizon_interface_indices for train and test pieces
            train_coords_k = unit.coords[train_index]
            train_data_k = unit.unit_data[train_index]
            test_coords_k = unit.coords[test_index]
            test_data_k = unit.unit_data[test_index]

            if unit.unit_dict is None:
                unit_k_train = UnitData(train_coords_k, train_data_k)
                unit_k_train.unit_vtk = create_vtk_polydata_from_coords_and_property(unit_k_train.coords,
                                                                                     unit_k_train.unit_data,
                                                                                     "unit")
                unit_k_test = UnitData(test_coords_k, test_data_k)
                unit_k_test.unit_vtk = create_vtk_polydata_from_coords_and_property(unit_k_test.coords,
                                                                                    unit_k_test.unit_data,
                                                                                    "unit")
            else:
                unit_k_train = UnitData(train_coords_k, train_data_k, unit.unit_id_to_unit_level)
                unit_k_train.unit_vtk = create_vtk_polydata_from_coords_and_property(unit_k_train.coords,
                                                                                     unit_k_train.unit_data,
                                                                                     "unit")
                unit_k_test = UnitData(test_coords_k, test_data_k, unit.unit_id_to_unit_level)
                unit_k_test.unit_vtk = create_vtk_polydata_from_coords_and_property(unit_k_test.coords,
                                                                                    unit_k_test.unit_data,
                                                                                    "unit")
            self.split.append((unit_k_train, unit_k_test))