import os
import pandas as pd
import numpy as np
import torch
import pyvista as pv
from geoinr.input.readers import reader_xml_polydata_file
from geoinr.utils.vtk_utils import add_np_property_to_vtk_object, create_vtk_polydata_from_coords_and_property
from vtkmodules.all import vtkPolyData
from sklearn.model_selection import KFold


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


class Series(object):
    def __init__(self, interface_info_file):
        """
        :param interface_info_file: csv file
            index formation_name unit_code relation
            0     rock a               12  erosional
            1     rock b               13  onlap
            2     rock c               14  onlap
            3     rock d               15  onlap
            4     rock e               16  erosional
            5     rock f               17  erosional
            6     rock g               18  erosional
        unit_code = 12 younger than unit_code = 13. unit_code 18 is oldest. unit_code 12 is youngest.
        actual unit_code value not meaning, the order is what is meaningful. and captures the relevant info
        """
        # read csv file containing series/interface info
        self.series_info_df = pd.read_csv(interface_info_file)
        self.n_horizons = len(self.series_info_df.index)
        self.n_unit_classes = self.n_horizons + 1
        self.horizon_level_code = self.series_info_df['fm_code'].to_dict()
        self.unit_level_code = self.build_unit_level_code()

        # build horizon indices for each series and create tuple (horizon_indices, relationship)
        # for each stored in a list
        series = []
        onlap_horizon_indices = []
        for index, row in self.series_info_df.iterrows():
            if row['relationship'] == 'erosional':
                if onlap_horizon_indices:
                    series.append((onlap_horizon_indices, 'onlap'))
                    onlap_horizon_indices = []  # reset
                series.append((index, 'erosional'))
            if row['relationship'] == 'onlap':
                onlap_horizon_indices.append(index)
        if onlap_horizon_indices:
            series.append((onlap_horizon_indices, 'onlap'))
        self.series = series
        self.n_series = len(self.series)
        self.mean_scalar_values_for_series = None
        self.series_dict = None  # key = series_id, value = list of horizon indices associated with series
        self.unconformity_dict = None  # key = series_id, value = horizon index associated with series
        self.unit_dict = None  # key = series_id, value = list of unit_ids associated with series
        self.series_df = None
        self.series_cutting_sequence = None
        self.cutting_series_is_onlap = None
        self.above_below_horizons_and_series_for_units = None
        self.above_below_horizons_and_series_for_horizons = None
        self.build_dictionaries_and_dataframe()
        self.generate_sequence_for_series_cutting()
        self.build_geological_domains()
        self.generate_above_below_horizons_and_series_for_units()
        self.generate_above_below_horizons_and_series_for_horizons()
        self.build_unit_dictionary()
        t = 6

    def build_dictionaries_and_dataframe(self):
        series_dict = {}
        unconformity_dict = {}
        relations = []
        horizon_indices = []
        series_id = []
        for index, (horizon_i_indices, horizon_i_relation) in enumerate(self.series):
            series_id.append(index)
            relations.append(horizon_i_relation)
            if horizon_i_relation == 'erosional':
                unconformity_dict[index] = horizon_i_indices
            if not isinstance(horizon_i_indices, list):
                horizon_i_indices = [horizon_i_indices]
            horizon_indices.append(horizon_i_indices)
            series_dict[index] = horizon_i_indices
        interface_s_id = {}
        for s_id, interface_indices in series_dict.items():
            for interface_index in interface_indices:
                interface_s_id[interface_index] = s_id
        interface_unit_above = {}
        interface_unit_below = {}
        for i, interface_index in enumerate(interface_s_id.keys()):
            interface_unit_above[interface_index] = i
            interface_unit_below[interface_index] = i + 1
        self.series_info_df['series_id'] = self.series_info_df.index.map(interface_s_id)
        self.series_info_df['unit_above'] = self.series_info_df.index.map(interface_unit_above)
        self.series_info_df['unit_below'] = self.series_info_df.index.map(interface_unit_below)
        self.series_dict = series_dict
        self.unconformity_dict = unconformity_dict
        self.series_df = pd.DataFrame({'s_id': series_id,
                                       'horizon_indices': horizon_indices,
                                       'relationship': relations})

    def build_unit_level_code(self):
        unit_level_code = {}
        for i, (horizon_id, horizon_level_code) in enumerate(self.horizon_level_code.items()):
            unit_level_code[i] = horizon_level_code - 1
        unit_level_code[self.n_unit_classes - 1] = self.horizon_level_code[self.n_horizons - 1]
        return unit_level_code

    def build_geological_domains(self):

        domain_below_interfaces_idxs = []
        below_interface_idxs = []  # temp container
        domain_idxs = []
        domain_idx = 0
        for interface_idx in self.series_info_df.index:
            below_interface_idxs.append(interface_idx)
            domain_idxs.append(domain_idx)
            if self.series_info_df['relationship'][interface_idx] == 'erosional':
                domain_below_interfaces_idxs.append(below_interface_idxs)
                below_interface_idxs = []
                domain_idx += 1
        self.series_info_df['domain_idx'] = domain_idxs
        self.geo_domain_info_df = pd.DataFrame.from_dict({'below_interface_idxs': domain_below_interfaces_idxs})

    def build_unit_dictionary(self):
        """-
        builds a dictionary associating unit ids to specific series ids
        unit ids are build younger to older; older unit ids are larger than younger units
        key = series id
        value = list of unit ids
        """
        unit_dict = {i: [] for i in range(self.n_series)}
        unit_id = 0

        for i in range(self.series_cutting_sequence.size):
            series_id = self.series_cutting_sequence[i]
            if self.cutting_series_is_onlap[i]:
                for j in range(len(self.series_dict[series_id])):
                    unit_dict[series_id].append(unit_id)
                    unit_id += 1
                # add unit id for below last onlap series horizon
                unit_dict[series_id].append(unit_id)
                unit_id += 1
            else:
                unit_dict[series_id].append(unit_id)
                unit_id += 1
        # add unit id for below basement
        unit_dict[self.n_series - 1].append(unit_id)
        self.unit_dict = unit_dict

    def get_unconformity_series_ids_younger_than(self, series_id):
        younger_unconformity_df = self.series_df.loc[(self.series_df['relationship'] == 'erosional') &
                                                     (self.series_df['s_id'] < series_id)]
        younger_unconformity_series_id = younger_unconformity_df['s_id'].values
        # return s_ids from older(large) to younger(smaller)
        return younger_unconformity_series_id[::-1]

    def get_unconformity_series_ids(self):
        unconformity_df = self.series_df.loc[self.series_df['relationship'] == 'erosional']
        return unconformity_df['s_id'].values

    def get_onlap_series_ids(self):
        onlap_df = self.series_df.loc[self.series_df['relationship'] == 'onlap']
        return onlap_df['s_id'].values

    def get_unconformity_series_ids_below(self, series_id):
        unconformity_df = self.series_df.loc[(self.series_df['relationship'] == 'erosional') &
                                             (self.series_df['s_id'] > series_id)]
        older_unconformity_series_id = unconformity_df['s_id'].values
        return older_unconformity_series_id

    def set_mean_scalar_values_for_series(self, series_mean_scalar_values: torch.Tensor):
        self.mean_scalar_values_for_series = series_mean_scalar_values

    def generate_sequence_for_series_cutting(self):
        """
        This function determines which series to cut. A series that is cut will generate a domain in which is valid
        for series. This information is used in a sequential manner to respect the geological history. Processed
        younger to older.
        The data structures built here optimize the cutting methods used later.
        Example:
        Series {0: [0], 1: [1, 2, 3], 2: [4], 3: [5], 4: [6] }, Unconformity {0: 0, 2: 4, 3: 5, 4: 6}
        key = series id, value = horizon indices

        series_cutting_sequence: [0, 1, 3, 4] (series ids)
                         cut_by: [0, 2, 3, 4] (unconformity series ids)
        cutting_series_is_onlap: [False, True, False, False]
        NOTE: Series 2 is NOT cut b/c we use this series ONLY to cut the onlap series deposited on top of 2. We don't
              need it to assigned units or record resultant scalar fields.
        Encodes series_cutting_sequence (integer array) AND cutting_series_is_onlap (boolean array)
        """
        unconformity_series_ids = self.get_unconformity_series_ids()

        series_id = [0]
        for i in range(unconformity_series_ids.size - 1):
            series_id.append(unconformity_series_ids[i] + 1)
        self.series_cutting_sequence = np.array(series_id)

        onlap_series_ids = self.get_onlap_series_ids()
        is_onlap = []
        for domain_index in self.series_cutting_sequence:
            if np.isin(domain_index, onlap_series_ids):
                is_onlap.append(True)
            else:
                is_onlap.append(False)
        self.cutting_series_is_onlap = np.array(is_onlap)

    def generate_above_below_horizons_and_series_for_units(self):
        above_below_horizons_and_series = {}
        for unit_id in range(self.n_unit_classes):
            unit_bounded_horizons = {}
            if unit_id == 0:
                bounded_horizons = (None, unit_id)
            elif unit_id == (self.n_unit_classes - 1):
                bounded_horizons = (unit_id - 1, None)
            else:
                bounded_horizons = (unit_id - 1, unit_id)
            # get series associated with top horizon
            above_horizons = None
            above_series = None
            if bounded_horizons[0] is not None:
                # get horizons above and including bounded_horizon[0]
                above_horizons = np.arange(bounded_horizons[0] + 1).tolist()
                # find series id associated with above horizons
                above_series = []
                for above_horizon in above_horizons:
                    for series_id, horizon_ids in self.series_dict.items():
                        if above_horizon in horizon_ids:
                            above_series.append(series_id)
                            break
            below_horizons = None
            below_series = None
            if bounded_horizons[1] is not None:
                # find the interfaces below unit_id within the geological domain associated with unit
                # get geological domain index
                domain_idx = self.series_info_df.iloc[unit_id]['domain_idx']  # unit_id == below horizon id
                below_horizons = [interface_idx for interface_idx in
                                  self.geo_domain_info_df.iloc[domain_idx]['below_interface_idxs'] if
                                  interface_idx >= unit_id]
                if below_horizons:
                    below_series = []
                    for below_horizon in below_horizons:
                        for series_id, horizon_ids in self.series_dict.items():
                            if below_horizon in horizon_ids:
                                below_series.append(series_id)
                                break
                else:
                    below_horizons = None

            above_below_horizons_and_series[unit_id] = {'above_horizons': above_horizons,
                                                        'above_series': above_series,
                                                        'below_horizons': below_horizons,
                                                        'below_series': below_series}

        self.above_below_horizons_and_series_for_units = above_below_horizons_and_series

    def generate_above_below_horizons_and_series_for_horizons(self):
        # Go from youngest to oldest
        above_below_horizons_and_series = {}
        for horizon_id in range(self.n_horizons):
            bounded_horizons = {}
            if horizon_id == 0:
                bounded_horizons = (None, horizon_id + 1)
            elif horizon_id == (self.n_horizons - 1):
                bounded_horizons = (horizon_id - 1, None)
            else:
                bounded_horizons = (horizon_id - 1, horizon_id + 1)
            # get series associated with top horizon
            above_horizons = None
            above_series = None
            if bounded_horizons[0] is not None:
                # get horizons above and including bounded_horizon[0]
                above_horizons = np.arange(bounded_horizons[0] + 1).tolist()
                # find series id associated with above horizons
                above_series = []
                for above_horizon in above_horizons:
                    for series_id, horizon_ids in self.series_dict.items():
                        if above_horizon in horizon_ids:
                            above_series.append(series_id)
                            break
            below_horizons = None
            below_series = None
            if bounded_horizons[1] is not None:
                # find the interfaces below horizon_id within the geological domain associated with it
                # get geological domain index
                domain_idx = self.series_info_df.iloc[horizon_id]['domain_idx']
                below_horizons = [interface_idx for interface_idx in self.geo_domain_info_df.iloc[domain_idx]['below_interface_idxs'] if interface_idx > horizon_id]
                if below_horizons:
                    below_series = []
                    for below_horizon in below_horizons:
                        for series_id, horizon_ids in self.series_dict.items():
                            if below_horizon in horizon_ids:
                                below_series.append(series_id)
                                break
                else:
                    below_horizons = None

            above_below_horizons_and_series[horizon_id] = {'above_horizons': above_horizons,
                                                           'above_series': above_series,
                                                           'below_horizons': below_horizons,
                                                           'below_series': below_series}
        above_below_interfaces_for_interfaces = pd.DataFrame.from_dict(above_below_horizons_and_series, orient='index')
        self.series_info_df = self.series_info_df.join(above_below_interfaces_for_interfaces, lsuffix='_caller', rsuffix='_other')
        self.above_below_horizons_and_series_for_horizons = above_below_horizons_and_series


class InterfaceData(object):
    def __init__(self, *args):
        """Dataset manager for interface point data.
        Each 3D point has an [x, y, z] and a scalar value associated with that specifies where in the stratigraphic
        sequence it exists. Higher values are younger.
        Dataset contains:
        coords: a matrix of 3D coords [N, 3]
        interface_data: vector of scalar values [N, 1] / [1, N]. interface_data[0] is associated with point coords[0]
        horizon_interface_indices: a list of point indices (a list) for each unique sampled horizon - list of lists
                                   [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11]]
                                   horizon_interface_indices[0] = [0, 1, 2, 3, 4] point indices for youngest horizon
                                   horizon_interface_indices[2] = [9, 10, 11] point indices for oldest horizon
        coords scaling/normalization occurs after dataset created, when calling self.transform(scalar), where scalar
        is a sklearn scaling function (StandardScalar, MinMaxScalar). Occurs after since the scaling function can only
        be setup after all datasets are loading and a modelling domain bounds using those datasets are computed/defined
        Three types of encoded interface data level_data_mode:
        1) level properties are in the correct order (largest numbers (youngest units) -> smallest numbers (oldest)),
           not normalized
        2) reverse level properties (smallest numbers (youngest units) -> largest numbers (oldest))
           not normalized
        3) level properties are in the correct order and IS normalized.
        """
        self.interface_vtk = None
        self.interface_bounds = None
        self.interface_df = None
        self.n_interface = 0
        self.n_horizons = 0
        self.coords = None
        self.interface_data = None
        self.level_data_mode = 1
        self.series = None
        self.unique_interface_values = None
        self.horizon_weights = None
        self.interface_weights = None
        self.strat_rel_weights = None
        self.interface_horizon_index = None
        self.horizon_interface_indices = None
        self.scalar_values_for_strat_sequence = None
        self.class_units_for_strat_sequence = None
        self.boundary_prob = None
        self.scalar = None
        self.original_interface_indices = None

        # Properties computed from modelling
        self.scalar_pred = None
        self.residuals = None
        self.grad_norm_pred = None
        self.horizon_scalar_means = None
        self.horizon_residual_means = None
        self.horizon_residual_std = None
        self.horizon_variance = None
        self.interface_metric_dict = None
        self.horizon_dist_residuals = None
        self.horizon_dist_std = None

        if len(args) >= 2:
            if len(args) == 2:
                assert isinstance(args[0], str), "For InterfaceData with 2 argument initializer, the first argument " \
                                                 "must be a string"
                # vtk file with level_data_mode. Not a multiple series.
                self.build_data_from_files(args[0], args[1])
            elif len(args) == 3:
                if isinstance(args[0], str):
                    self.build_data_from_files(args[0], args[1], args[2])
                else:
                    self.build_from_data_and_level_data_mode(args[0], args[1], args[2])
            else:
                raise IOError("Unexpected number of inputted arguments into InterfaceData constructor")
        else:
            raise IOError("Unexpected number of inputted arguments into InterfaceData constructor")
        t = 7

    def build_data_from_files(self, interface_file: str, level_data_mode: int, interface_info_file=None):
        # Read VTK file containing interface data
        if not os.path.isfile(interface_file):
            raise ValueError('File for interface data does not exist')
        self.interface_vtk = reader_xml_polydata_file(interface_file)
        if interface_info_file is not None:
            if not os.path.isfile(interface_info_file):
                raise ValueError('File for interface info (series) does not exist')
            self.series = Series(interface_info_file)
        # Get data bounds [xmin, xmax, ymin, ymax, zmin, zmax]
        self.interface_bounds = self.interface_vtk.GetBounds()
        # Create pandas dataframe
        self.interface_df = convert_vtkPolydata_contact_to_dataframe(self.interface_vtk)
        coords = self.interface_df[['x', 'y', 'z']].to_numpy()
        # Get interface data (stratigraphically sequenced scalar values s).
        # Larger s means younger, smaller s means older
        interface_data = self.interface_df[['level']].to_numpy()
        self.build_from_data_and_level_data_mode(coords, interface_data, level_data_mode)

    def build_from_data_and_level_data_mode(self, coords: np.ndarray, level_data: np.ndarray, level_data_mode: int):
        self.level_data_mode = level_data_mode
        self.coords = coords
        interface_data = level_data
        # Get interface data (stratigraphically sequenced scalar values s).
        # Larger s means younger, smaller s means older
        self.interface_data, self.unique_interface_values, \
        counts = self.extract_interface_constraints(interface_data, level_data_mode)
        self.n_interface = self.interface_data.size
        self.n_horizons = self.unique_interface_values.size
        # Get weights related to interface data
        max_count = counts.max()
        self.horizon_weights = max_count / counts
        interface_weight_map = {self.unique_interface_values[i]: self.horizon_weights[i]
                                for i in range(self.n_horizons)}
        self.interface_weights = np.vectorize(interface_weight_map.get)(self.interface_data)
        # get link between an interface point and the ordered horizon index
        # e.g. for point idx 3, interface_horizon_index will encode its link to a specific horizon 0
        #      among possible indices [0 (youngest), 1, 2, 3 (oldest)]
        interface_horizon_map = {self.unique_interface_values[i]: i for i in range(self.n_horizons)}
        self.interface_horizon_index = np.vectorize(interface_horizon_map.get)(self.interface_data)
        # Create list of interfaces indices for each unique horizon/interface. List of lists. First is the
        # youngest interface, while the last is the oldest.
        self.horizon_interface_indices = {interface_scalar_value: []
                                          for interface_scalar_value in self.unique_interface_values}
        for i in range(self.n_interface):
            self.horizon_interface_indices[self.interface_data[i]].append(i)
        self.horizon_interface_indices = [interface_indices
                                          for interface_indices in self.horizon_interface_indices.values()]

    def extract_interface_constraints(self, interface_data, mode, s_min=-1.0, s_max=1.0):
        """
        Three modes of extracting scalar constraints for interface data:
        1) level properties are in the correct order (largest numbers (youngest units) -> smallest numbers (oldest)),
           not normalized
        2) reverse level properties (smallest numbers (youngest units) -> largest numbers (oldest))
           not normalized
        3) level properties are in the correct order and IS normalized.
        return: interface_data (scalar constraints)
                unique_interface_values (ordered youngest->oldest, largest value->smallest value)
                counts (number of constraints per unique interface value, ordered consistent with unique values)
        """
        assert mode in [1, 2, 3], "unknown mode for extracting interface constraints"

        unique_interface_values, counts = np.unique(interface_data, return_counts=True)
        n_horizons = unique_interface_values.size
        uniform_interface_values = np.linspace(s_min, s_max, n_horizons)
        if mode == 1:
            interface_value_map = {unique_interface_values[i]: uniform_interface_values[i]
                                   for i in range(n_horizons)}
            interface_data = np.vectorize(interface_value_map.get)(interface_data)
            unique_interface_values = uniform_interface_values
            counts = counts[::-1]  # flip counts to represent the counts for youngest->oldest
        elif mode == 2:
            interface_value_map = {unique_interface_values[i]: uniform_interface_values[(n_horizons - 1) - i]
                                   for i in range(n_horizons)}
            interface_data = np.vectorize(interface_value_map.get)(interface_data)
            unique_interface_values = uniform_interface_values
            # no need to flip counts for this one because this mode the unique values already represent
            # youngest->oldest
        else:  # mode == 3
            counts = counts[::-1]  # flip counts to represent the counts for youngest->oldest
        interface_data = interface_data.flatten()
        unique_interface_values = unique_interface_values[::-1]
        return interface_data, unique_interface_values.copy(), counts

    def get_coords(self):
        return self.coords

    def transform(self, scalar):
        self.scalar = scalar
        self.coords = scalar.transform(self.coords)

    def convert_to_torch(self):
        self.coords = torch.from_numpy(self.coords).float()
        self.coords = self.coords.contiguous()
        self.interface_data = torch.from_numpy(self.interface_data).float()
        self.interface_weights = torch.from_numpy(self.interface_weights).float()
        self.horizon_interface_indices = [torch.LongTensor(horizon_i_indices)
                                          for horizon_i_indices in self.horizon_interface_indices]
        self.interface_horizon_index = torch.from_numpy(self.interface_horizon_index).long()
        self.unique_interface_values = torch.from_numpy(self.unique_interface_values).float()
        self.horizon_weights = torch.from_numpy(self.horizon_weights).float()

    def send_to_gpu(self, rank):
        self.coords = self.coords.to(rank)
        self.interface_data = self.interface_data.to(rank)
        self.interface_weights = self.interface_weights.to(rank)
        self.horizon_interface_indices = [horizon_i_indices.to(rank)
                                          for horizon_i_indices in self.horizon_interface_indices]
        self.interface_horizon_index = self.interface_horizon_index.to(rank)
        self.unique_interface_values = self.unique_interface_values.to(rank)
        self.horizon_weights = self.horizon_weights.to(rank)

    def set_coords(self, coords):
        self.coords = coords

    def set_interface_data(self, interface_data):
        self.interface_data = interface_data

    def set_interface_weights(self, interface_weights):
        self.interface_weights = interface_weights

    def set_horizon_interface_indices(self, horizon_interface_indices):
        self.horizon_interface_indices = horizon_interface_indices

    def set_interface_horizon_indices(self, interface_horizon_indices):
        self.interface_horizon_index = interface_horizon_indices

    def set_original_interface_indices(self, original_interface_indices):
        self.original_interface_indices = original_interface_indices

    def set_scalar_pred(self, scalar_pred):
        self.scalar_pred = scalar_pred

    def set_residuals(self, residuals):
        self.residuals = residuals

    def set_grad_norm_pred(self, grad_norm_pred):
        self.grad_norm_pred = grad_norm_pred

    def set_horizon_scalar_means(self, horizon_scalar_means):
        self.horizon_scalar_means = horizon_scalar_means

    def set_horizon_residual_means(self, horizon_residual_means):
        self.horizon_residual_means = horizon_residual_means

    def set_horizon_residual_std(self, horizon_residual_std):
        self.horizon_residual_std = horizon_residual_std

    def set_horizon_variance(self, horizon_variance):
        self.horizon_variance = horizon_variance

    def set_horizon_dist_residuals(self, horizon_dist_residuals):
        self.horizon_dist_residuals = horizon_dist_residuals

    def set_horizon_dist_std(self,  horizon_dist_std):
        self.horizon_dist_std = horizon_dist_std

    def __generate_horizon_metric_dict(self):
        horizon_metric_dict = {}
        if self.horizon_scalar_means is not None:
            horizon_metric_dict['scalar_means'] = self.horizon_scalar_means
        if self.horizon_residual_means is not None:
            horizon_metric_dict['residual_means'] = self.horizon_residual_means
        if self.horizon_residual_std is not None:
            horizon_metric_dict['residual_std'] = self.horizon_residual_std
        if self.horizon_variance is not None:
            horizon_metric_dict['variance'] = self.horizon_variance
        if self.horizon_dist_residuals is not None:
            horizon_metric_dict['dist_residuals'] = self.horizon_dist_residuals
        if self.horizon_dist_std is not None:
            horizon_metric_dict['dist_std'] = self.horizon_dist_std
        if horizon_metric_dict:
            horizon_metric_df = pd.DataFrame(horizon_metric_dict)
            return {'horizon_metrics': horizon_metric_df}
        else:
            return None

    def __add_properties_to_vtk_object_if_present(self):
        added_properties = False
        if self.scalar_pred is not None:
            added_properties = True
            if isinstance(self.interface_vtk, pv.DataSet):
                self.interface_vtk.point_data["scalar_field"] = self.scalar_pred
            else:
                add_np_property_to_vtk_object(self.interface_vtk, "scalar_field", self.scalar_pred)
        if self.residuals is not None:
            added_properties = True
            if isinstance(self.interface_vtk, pv.DataSet):
                self.interface_vtk.point_data["residuals"] = self.residuals
            else:
                add_np_property_to_vtk_object(self.interface_vtk, "residuals", self.residuals)
        if self.grad_norm_pred is not None:
            added_properties = True
            if isinstance(self.interface_vtk, pv.DataSet):
                self.interface_vtk.point_data["grad_norm"] = self.grad_norm_pred
            else:
                add_np_property_to_vtk_object(self.interface_vtk, "grad_norm", self.grad_norm_pred)
        if added_properties:
            return self.interface_vtk
        else:
            return None

    def process_model_outputs(self):
        # 1) generate interface metric dictionary on model results data
        self.interface_metric_dict = self.__generate_horizon_metric_dict()
        # 2) add model results to vtk object
        self.interface_vtk = self.__add_properties_to_vtk_object_if_present()

    def __len__(self):
        return self.n_interface

    def __getitem__(self, idx):
        return self.coords[idx], self.interface_data[idx], self.interface_weights[idx]


class InterfaceKFoldSplit(object):
    def __init__(self, interface: InterfaceData, k, shuffle=False):
        self.split = []  # each element will be an InterfaceData object
        kf = KFold(n_splits=k, shuffle=shuffle)
        for train_index, test_index in kf.split(interface.coords):
            # have to re-index horizon_interface_indices for train and test pieces
            train_coords_k = interface.coords[train_index]
            train_data_k = interface.interface_data[train_index]
            test_coords_k = interface.coords[test_index]
            test_data_k = interface.interface_data[test_index]

            interface_k_train = InterfaceData(train_coords_k, train_data_k, 3)
            interface_k_train.interface_vtk = create_vtk_polydata_from_coords_and_property(interface_k_train.coords,
                                                                                           interface_k_train.interface_data,
                                                                                           "level")


            interface_k_train.series = interface.series
            interface_k_test = InterfaceData(test_coords_k, test_data_k, 3)
            interface_k_test.interface_vtk = create_vtk_polydata_from_coords_and_property(interface_k_test.coords,
                                                                                          interface_k_test.interface_data,
                                                                                          "level")
            self.split.append((interface_k_train, interface_k_test))



