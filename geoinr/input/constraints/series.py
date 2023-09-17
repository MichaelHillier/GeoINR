import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sn


class GeologicalDomains(object):
    def __init__(self, geo_domain_info: pd.DataFrame):
        # We reverse the below lists because we process the domains older to younger (the df is listed younger to older)
        domain_indices = geo_domain_info['domain_index'].tolist()
        domain_indices.reverse()
        self.domain_indices = domain_indices
        self.n_domains = len(self.domain_indices)
        above_boundary_interface = geo_domain_info['above_boundary_surf'].tolist()
        above_boundary_interface.reverse()
        self.above_boundary_interface = above_boundary_interface
        above_boundary_interface_s_id = geo_domain_info['above_boundary_surf_id'].tolist()
        above_boundary_interface_s_id.reverse()
        self.above_boundary_interface_s_id = above_boundary_interface_s_id
        conformal_interfaces = geo_domain_info['conformal_interfaces'].tolist()
        for conformal_set_of_interfaces in conformal_interfaces:
            if conformal_set_of_interfaces:
                conformal_set_of_interfaces.reverse()
        conformal_interfaces.reverse()
        self.conformal_interfaces = conformal_interfaces
        units = geo_domain_info['units'].tolist()
        for unit_set in units:
            # if unit_set is increasing in index, reverse it
            is_increasing = all(x < y for x, y in zip(unit_set, unit_set[1:]))
            if is_increasing:
                unit_set.reverse()
        units.reverse()
        self.units = units
        series_id = geo_domain_info['series_id'].tolist()
        series_id.reverse()
        self.series_id = series_id
        cutting_boundaries = geo_domain_info['cutting_boundaries'].tolist()
        cutting_boundaries.reverse()
        self.cutting_boundaries = cutting_boundaries
        cutting_boundaries_s_id = geo_domain_info['cutting_boundaries_s_id'].tolist()
        cutting_boundaries_s_id.reverse()
        self.cutting_boundaries_s_id = cutting_boundaries_s_id


class Series(object):
    def __init__(self, interface_info_file, efficient=True, has_youngest_unit_sampled=False):
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
        self.has_youngest_unit_sampled = has_youngest_unit_sampled
        self.unit_level_code = self.build_unit_level_code()

        # build horizon indices for each series and create tuple (horizon_indices, relationship)
        # for each stored in a list
        series = []
        onlap_horizon_indices = []
        baselap_horizon_indices = []
        basement_horizon_indices = []
        for index, row in self.series_info_df.iterrows():
            if row['relationship'] == 'erosional':
                if onlap_horizon_indices:
                    series.append((onlap_horizon_indices, 'onlap'))
                    onlap_horizon_indices = []  # reset
                if baselap_horizon_indices:
                    series.append((baselap_horizon_indices, 'baselap'))
                    baselap_horizon_indices = []  # resset
                series.append((index, 'erosional'))
            if row['relationship'] == 'baselap':
                if baselap_horizon_indices:
                    series.append((baselap_horizon_indices, 'baselap'))
                    baselap_horizon_indices = [index]
                else:
                    baselap_horizon_indices.append(index)
            if row['relationship'] == 'onlap':
                if baselap_horizon_indices:
                    baselap_horizon_indices.append(index)
                else:
                    onlap_horizon_indices.append(index)
            if row['relationship'] == 'basement':
                basement_horizon_indices.append(index)
        if onlap_horizon_indices:
            series.append((onlap_horizon_indices, 'onlap'))
        if baselap_horizon_indices:
            series.append((baselap_horizon_indices, 'baselap'))
        if basement_horizon_indices:
            series.append((basement_horizon_indices, 'basement'))
        self.series = series
        self.n_series = len(self.series)
        self.series_indices = np.arange(self.n_series).tolist()
        self.mean_scalar_values_for_series = None
        self.series_dict = None  # key = series_id, value = list of horizon indices associated with series
        self.unconformity_dict = None  # key = series_id, value = horizon index associated with series
        self.boundary_surf_dict = None  # key = series_id, value = horizon index associated with boundary surface
        self.boundary_surf = None  # list of info for boundary surfaces (unconform, baselap).
        # info is tuples (series_id, interface_id)
        self.series_df = None
        self.geo_domain_info_df = None
        self.geological_domains = None
        self.above_below_horizons_and_series_for_units = None
        self.above_below_horizons_and_series_for_horizons = None
        self.unique_s_ids_for_interfaces_above_below = None
        self.above_below_info = None
        self.unique_s_ids_for_units_above_below = None
        self.above_below_unit_info = None
        self.build_dictionaries_and_dataframe()
        self.build_geological_domains()
        self.generate_above_below_horizons_and_series_for_horizons(efficient=efficient)
        self.generate_above_below_horizons_and_series_for_units()
        t = 6

    def build_dictionaries_and_dataframe(self):
        series_dict = {}
        unconformity_dict = {}
        boundary_surf_dict = {}
        relations = []
        horizon_indices = []
        series_id = []
        for index, (horizon_i_indices, horizon_i_relation) in enumerate(self.series):
            series_id.append(index)
            relations.append(horizon_i_relation)
            if horizon_i_relation == 'erosional':
                unconformity_dict[index] = horizon_i_indices
            if horizon_i_relation == 'erosional' or horizon_i_relation == 'baselap' or horizon_i_relation == 'basement':
                if not isinstance(horizon_i_indices, list):
                    boundary_surf_dict[index] = horizon_i_indices
                else:
                    # if a series contains a baselap boundary surf, it's always the first one in the list
                    boundary_surf_dict[index] = horizon_i_indices[0]
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
        self.series_info_df['interface_id'] = self.series_info_df.index
        self.series_info_df['series_id'] = self.series_info_df.index.map(interface_s_id)
        self.series_info_df['unit_above'] = self.series_info_df.index.map(interface_unit_above)
        self.series_info_df['unit_below'] = self.series_info_df.index.map(interface_unit_below)
        self.series_dict = series_dict
        self.unconformity_dict = unconformity_dict
        self.boundary_surf_dict = boundary_surf_dict
        boundary_surf = [(series_id, interface_id) for series_id, interface_id in boundary_surf_dict.items()]
        boundary_surf.reverse()  # older to younger. list of tuples (series_id, interface_id)
        self.boundary_surf = boundary_surf
        self.series_df = pd.DataFrame({'s_id': series_id,
                                       'horizon_indices': horizon_indices,
                                       'relationship': relations})

    def build_unit_level_code(self):
        # TODO - this code is very error prone depending on inputted datasets.
        # Need to make it more robust
        # For now, we will use an argument flag: --has_youngest_unit_sampled
        unit_level_code = {}
        if not self.has_youngest_unit_sampled:
            unit_level_code[0] = self.horizon_level_code[0] - 1
        for i, (horizon_id, horizon_level_code) in enumerate(self.horizon_level_code.items()):
            if self.has_youngest_unit_sampled:
                unit_level_code[i] = horizon_level_code
            else:
                unit_level_code[i + 1] = horizon_level_code
        if self.has_youngest_unit_sampled:
            unit_level_code[self.n_unit_classes - 1] = self.horizon_level_code[self.n_horizons - 1] + 1
        return unit_level_code

    def build_geological_domains(self):
        #    Determines for each domain:
        #    above_boundary_surface (the boundary surface directly/immediately above the domain!)
        #    above_boundary_surface_s_id
        #    conformal_interfaces = [] list of conformal interface within the domain
        #    units = [] list of units within the domain. If it has conformal interface, there is more than one. Else
        #               there is one
        #    scalar_field: the field associated within the domain (can only be one). Note a domain can be defined using
        #                  multiple boundary interfaces and associated scalar fields
        #    cutting_boundaries: the interfaces that sequentially cut the above_boundary_surface
        #    cutting_boundaries_s_id: series id's for each of the cutting boundaries (in the correct sequence)

        # build comprehensive geological domain DF
        n_boundary_surf = len(self.boundary_surf)
        above_boundary_surface = []
        above_boundary_surf_s_id = []
        conformal_interfaces = []
        units = []
        domains = []
        scalar_field = []
        d_idx = 0
        for boundary_s_id, boundary_itr_id in self.boundary_surf:
            domains.append(d_idx)
            d_idx += 1
            above_boundary_surface.append(boundary_itr_id)
            above_boundary_surf_s_id.append(boundary_s_id)
            # are there any conformal interface below this boundary
            # look in series_info_df. Check if there are any interface below this one that are onlap. take all
            # interface up until the next boundary.
            interfaces_below_df = self.series_info_df[self.series_info_df['interface_id'] > boundary_itr_id]
            conformal_interfaces_below = []
            if not interfaces_below_df.empty:
                # iterate over rows in interfaces_below_df and append interface_id's with onlap relations. stop once
                # a relation other than onlap is encounters (erosional, baselap, basement)
                for index, row in interfaces_below_df.iterrows():
                    if row['relationship'] == 'onlap':
                        conformal_interfaces_below.append(row['interface_id'])
                    else:
                        break
            if conformal_interfaces_below:
                conformal_interfaces.append(conformal_interfaces_below)
                scalar_field.append(self.series_info_df.iloc[conformal_interfaces_below[0]]['series_id'])
            else:
                conformal_interfaces.append(None)
                scalar_field.append(boundary_s_id)
            # Determine units for domain
            domain_units = [self.series_info_df.iloc[boundary_itr_id]['unit_below']]
            for conformal_interface_id in conformal_interfaces_below:
                domain_units.append(self.series_info_df.iloc[conformal_interface_id]['unit_below'])
            units.append(domain_units)

        # for the final boundary surface (younger) we process the final domain
        above_boundary_surface.append(-1)
        above_boundary_surf_s_id.append(-1)
        domains.append(d_idx)
        n_boundary = len(self.boundary_surf)
        last_boundary_surface = self.boundary_surf[n_boundary - 1]
        last_boundary_interface_id = last_boundary_surface[1]
        interfaces_above_df = self.series_info_df[self.series_info_df['interface_id'] < last_boundary_interface_id]
        conformal_interfaces_above = []
        if not interfaces_above_df.empty:
            # iterate over rows in interfaces_above_df and append interface_id's with onlap relations
            for index, row in interfaces_above_df.iterrows():
                if row['relationship'] == 'onlap':
                    conformal_interfaces_above.append(row['interface_id'])
                else:
                    break
        if conformal_interfaces_above:
            conformal_interfaces.append(conformal_interfaces_above)
            scalar_field.append(self.series_info_df.iloc[conformal_interfaces_above[0]]['series_id'])
        else:
            conformal_interfaces.append(None)
            scalar_field.append(boundary_s_id)
        # Determine units for domain
        domain_units = [self.series_info_df.iloc[last_boundary_interface_id]['unit_above']]
        for conformal_interface_id in conformal_interfaces_above:
            domain_units.append(self.series_info_df.iloc[conformal_interface_id]['unit_above'])
        units.append(domain_units)

        above_boundary_surface.reverse()
        above_boundary_surf_s_id.reverse()
        conformal_interfaces.reverse()
        units.reverse()
        scalar_field.reverse()

        # find all the boundaries that cut (in sequence) each domain
        cutting_boundaries = []
        cutting_boundaries_s_id = []
        for i, above_boundary_interface in enumerate(above_boundary_surface):
            domain_cutting_boundaries = []
            domain_cutting_boundies_s_id = []
            if above_boundary_interface < 0 or \
                    self.series_info_df.iloc[above_boundary_interface]['relationship'] == 'basement':
                domain_cutting_boundaries = None
                domain_cutting_boundies_s_id = None
            else:
                # find all younger boundary interfaces
                younger_boundary_interfaces_df = self.series_info_df[
                    (self.series_info_df['interface_id'] < above_boundary_interface) &
                    (self.series_info_df['relationship'] == 'erosional')]
                if not younger_boundary_interfaces_df.empty:
                    for index, younger_boundary_interface in younger_boundary_interfaces_df.iterrows():
                        domain_cutting_boundaries.append(younger_boundary_interface['interface_id'])
                        domain_cutting_boundies_s_id.append(younger_boundary_interface['series_id'])
                else:
                    domain_cutting_boundaries = None
                    domain_cutting_boundies_s_id = None
            if domain_cutting_boundaries:
                domain_cutting_boundaries.reverse()
                domain_cutting_boundies_s_id.reverse()
            cutting_boundaries.append(domain_cutting_boundaries)
            cutting_boundaries_s_id.append(domain_cutting_boundies_s_id)

        self.geo_domain_info_df = pd.DataFrame({'domain_index': domains,
                                                'above_boundary_surf': above_boundary_surface,
                                                'above_boundary_surf_id': above_boundary_surf_s_id,
                                                'conformal_interfaces': conformal_interfaces,
                                                'units': units,
                                                'series_id': scalar_field,
                                                'cutting_boundaries': cutting_boundaries,
                                                'cutting_boundaries_s_id': cutting_boundaries_s_id})

        self.geological_domains = GeologicalDomains(self.geo_domain_info_df)

    def get_unconformity_series_ids_younger_than(self, series_id):
        younger_unconformity_df = self.series_df.loc[(self.series_df['relationship'] == 'erosional') &
                                                     (self.series_df['s_id'] < series_id)]
        younger_unconformity_series_id = younger_unconformity_df['s_id'].values
        # return s_ids from older(large) to younger(smaller)
        return younger_unconformity_series_id[::-1]

    def get_unconformity_idx_younger_than(self, interface_idx):
        younger_unconformity_df = self.series_info_df.loc[(self.series_info_df['relationship'] == 'erosional') &
                                                          (self.series_info_df['interface_id'] < interface_idx)]
        younger_unconformity_interface_idx = younger_unconformity_df['interface_id'].values
        # return interface indices from older(large) to younger(smaller)
        return younger_unconformity_interface_idx[::-1]

    def get_unconformity_series_ids(self):
        unconformity_df = self.series_df.loc[self.series_df['relationship'] == 'erosional']
        return unconformity_df['s_id'].values

    def get_onlap_series_ids(self):
        onlap_df = self.series_df.loc[self.series_df['relationship'] == 'onlap']
        return onlap_df['s_id'].values

    def get_basement_interface_id_and_series_id(self):
        basement_df = self.series_info_df[self.series_info_df['relationship'] == 'basement']
        interface_id = None
        series_id = None
        if not basement_df.empty:
            assert len(basement_df.index) == 1, "More than one basement interface. Can't have"
            interface_id = basement_df.iloc[0]['interface_id']
            series_id = basement_df.iloc[0]['series_id']
        return interface_id, series_id

    def get_unconformity_series_ids_below(self, series_id):
        unconformity_df = self.series_df.loc[(self.series_df['relationship'] == 'erosional') &
                                             (self.series_df['s_id'] > series_id)]
        older_unconformity_series_id = unconformity_df['s_id'].values
        return older_unconformity_series_id

    def get_unconformity_series_id_below_series_id_if_onlap(self, series_id):
        series_id_relationship = self.series_df.iloc[series_id]['relationship']
        if series_id_relationship == 'onlap' or series_id_relationship == 'baselap':
            below_unconformity_series_df = self.series_df[(self.series_df['s_id'] > series_id) &
                                                          (self.series_df['relationship'] == 'erosional')]
            if not below_unconformity_series_df.empty:
                below_unconformity_series_ids = []
                for index, row in below_unconformity_series_df.iterrows():
                    unconformity_series_id = row['s_id']
                    below_unconformity_series_ids.append(unconformity_series_id)
                below_unconformity_series_ids.reverse()
                return below_unconformity_series_ids
            else:
                return []
        else:
            return []

    def get_baselap_interface_ids_set_below_series_id_if_onlap(self, series_id):
        series_id_relationship = self.series_df.iloc[series_id]['relationship']
        # onlap/baselap series act the same way wrt to baselap boundaries below it
        if series_id_relationship == 'onlap' or series_id_relationship == 'baselap':
            below_baselap_series_df = self.series_df[(self.series_df['s_id'] > series_id) &
                                                     (self.series_df['relationship'] == 'baselap')]
            if not below_baselap_series_df.empty:
                below_baselap_interface_ids = []
                for index, row in below_baselap_series_df.iterrows():
                    horizon_indices = row['horizon_indices']
                    boundary_interface = horizon_indices[0]
                    below_baselap_interface_ids.append(boundary_interface)
                below_baselap_interface_ids.reverse()
                return below_baselap_interface_ids
            else:
                return []
        else:
            return []

    def get_boundary_ids_below_to_next_unc_for_s_id_if_onlap(self, s_id):
        series_id_relationship = self.series_df.iloc[s_id]['relationship']
        # onlap/baselap series act the same way wrt to baselap boundaries below it
        if series_id_relationship == 'onlap' or series_id_relationship == 'baselap':
            below_boundary_df = self.series_df[(self.series_df['s_id'] > s_id) &
                                               ((self.series_df['relationship'] == 'baselap') |
                                                (self.series_df['relationship'] == 'erosional'))]
            if not below_boundary_df.empty:
                below_boundary_ids = []
                for index, row in below_boundary_df.iterrows():
                    horizon_indices = row['horizon_indices']
                    boundary_interface = horizon_indices[0]
                    below_boundary_ids.append(boundary_interface)
                    if row['relationship'] == 'erosional':
                        break
                below_boundary_ids.reverse()
                return below_boundary_ids
            else:
                return []
        else:
            return []

    def set_mean_scalar_values_for_series(self, series_mean_scalar_values: torch.Tensor):
        self.mean_scalar_values_for_series = series_mean_scalar_values

    def generate_above_below_horizons_and_series_for_units(self):
        above_interfaces = []
        above_series = []
        below_interfaces = []
        below_series = []
        oldest_interface_idx = self.series_info_df.iloc[-1]['interface_id']
        for unit_idx in self.unit_level_code.keys():
            # get interface index below unit_idx: interface_idx_below = unit_idx
            interface_index_below = unit_idx
            interface_index_above = unit_idx - 1
            if unit_idx <= oldest_interface_idx:
                below_interfaces_for_unit = [interface_index_below]
                series_id_interface_below = self.series_info_df.iloc[interface_index_below]['series_id']
                below_series_id_for_unit = [series_id_interface_below]
                # get set of below_horizons of interface_index_below
                if unit_idx < oldest_interface_idx:
                    below_horizons = self.series_info_df.iloc[interface_index_below]['below_horizons']
                    if below_horizons:
                        for i, below_horizon_i in enumerate(below_horizons):
                            below_interfaces_for_unit.append(below_horizon_i)
                            below_series_id_for_unit.append(self.series_info_df.iloc[below_horizon_i]['series_id'])
                below_interfaces.append(below_interfaces_for_unit)
                below_series.append(below_series_id_for_unit)
            else:
                below_interfaces.append(None)
                below_series.append(None)
            if interface_index_above >= 0:
                above_interfaces_for_unit = [interface_index_above]
                series_id_interface_above = self.series_info_df.iloc[interface_index_above]['series_id']
                above_series_id_for_unit = [series_id_interface_above]
                above_horizons = self.series_info_df.iloc[interface_index_above]['above_horizons']
                if above_horizons:
                    for above_horizon_i in above_horizons:
                        above_interfaces_for_unit.append(above_horizon_i)
                        above_series_id_for_unit.append(self.series_info_df.iloc[above_horizon_i]['series_id'])
                above_interfaces_for_unit.sort()
                above_series_id_for_unit.sort()
                above_interfaces.append(above_interfaces_for_unit)
                above_series.append(above_series_id_for_unit)
            else:
                above_interfaces.append(None)
                above_series.append(None)

        above_below_horizons_and_series = {}
        unit_unique_s_ids = []
        above_below_unit_info = []
        for unit_idx in self.unit_level_code.keys():
            above_unique_series_id = np.unique(above_series[unit_idx])
            below_unique_series_id = np.unique(below_series[unit_idx])
            unit_idx_unique_s_ids = []
            if above_unique_series_id[0] is None:
                above_unique_series_id = None
            else:
                above_unique_series_id = above_unique_series_id.tolist()
                unit_idx_unique_s_ids.append(above_unique_series_id)
            if below_unique_series_id[0] is None:
                below_unique_series_id = None
            else:
                below_unique_series_id = below_unique_series_id.tolist()
                unit_idx_unique_s_ids.append(below_unique_series_id)
            if unit_idx_unique_s_ids:
                flat_list = [s_id for sub_unique_s_ids in unit_idx_unique_s_ids for s_id in sub_unique_s_ids]
                flat_list = list(set(flat_list))
                unit_unique_s_ids.append(flat_list)
                above_below_unit_info.append([above_interfaces[unit_idx], below_interfaces[unit_idx],
                                              above_series[unit_idx], below_series[unit_idx]])
            else:
                unit_unique_s_ids.append(None)
                above_below_unit_info.append([None, None, None, None])
            above_below_horizons_and_series[unit_idx] = {'above_horizons': above_interfaces[unit_idx],
                                                         'above_series': above_series[unit_idx],
                                                         'above_u_series': above_unique_series_id,
                                                         'below_horizons': below_interfaces[unit_idx],
                                                         'below_series': below_series[unit_idx],
                                                         'below_u_series': below_unique_series_id,
                                                         }

        self.unique_s_ids_for_units_above_below = unit_unique_s_ids
        self.above_below_unit_info = above_below_unit_info
        above_below_interfaces_for_units = pd.DataFrame.from_dict(above_below_horizons_and_series, orient='index')

        self.above_below_horizons_and_series_for_units = above_below_horizons_and_series

    def generate_above_below_horizons_and_series_for_horizons(self, efficient=True):
        below_interfaces = []
        below_series = []
        for interface_idx in self.series_info_df.index:
            cur_relation = self.series_info_df.iloc[interface_idx]['relationship']
            below_interfaces_for_horizon = []
            below_series_id_for_horizon = []
            if cur_relation in ['erosional', 'basement']:
                below_interfaces_for_horizon = None
                below_series_id_for_horizon = None
            else:
                older_interface_indices = np.arange(interface_idx + 1, self.n_horizons)
                for older_interface_index in older_interface_indices:
                    older_interface_relation = self.series_info_df.iloc[older_interface_index]['relationship']
                    if older_interface_relation == 'onlap':
                        below_interfaces_for_horizon.append(older_interface_index)
                    elif older_interface_relation == 'baselap':
                        below_interfaces_for_horizon.append(older_interface_index)
                        # add next oldest unconformity
                        older_unconformities_df = self.series_info_df[
                            (self.series_info_df['relationship'] == 'erosional') &
                            (self.series_info_df['interface_id'] > older_interface_index)]
                        if not older_unconformities_df.empty:
                            n_o_unc_index = older_unconformities_df.iloc[0]['interface_id']
                            below_interfaces_for_horizon.append(n_o_unc_index)
                        break
                    else:
                        # erosional or basement
                        below_interfaces_for_horizon.append(older_interface_index)
                        break
                if below_interfaces_for_horizon:
                    below_series_id_for_horizon = [self.series_info_df.iloc[idx]['series_id']
                                                   for idx in below_interfaces_for_horizon]
                else:
                    below_interfaces_for_horizon = None
                    below_series_id_for_horizon = None
            below_interfaces.append(below_interfaces_for_horizon)
            below_series.append(below_series_id_for_horizon)

        above_interfaces = []
        above_series = []
        for interface_idx in self.series_info_df.index[::-1]:
            cur_relation = self.series_info_df.iloc[interface_idx]['relationship']
            above_interfaces_for_horizon = []
            above_series_id_for_horizon = []
            # if cur_relation in ['erosional', 'baselap', 'basement']:
            # get younger unconformities
            younger_unconformities_df = self.series_info_df[(self.series_info_df['relationship'] == 'erosional') &
                                                            (self.series_info_df['interface_id'] < interface_idx)]
            if efficient:
                if not younger_unconformities_df.empty:
                    y_unc_indices = younger_unconformities_df['interface_id'].values
                    for y_unc_index in y_unc_indices:
                        above_interfaces_for_horizon.append(int(y_unc_index))
                if cur_relation == 'onlap':
                    younger_interface_indices = np.arange(interface_idx)[::-1]
                    for younger_interface_index in younger_interface_indices:
                        younger_interface_relation = self.series_info_df.iloc[younger_interface_index]['relationship']
                        if younger_interface_relation == 'onlap':
                            above_interfaces_for_horizon.append(younger_interface_index)
                        elif younger_interface_relation == 'baselap':
                            above_interfaces_for_horizon.append(younger_interface_index)
                            younger_unconformities_df = self.series_info_df[
                                (self.series_info_df['relationship'] == 'erosional') &
                                (self.series_info_df['interface_id'] < younger_interface_index)]
                            if not younger_unconformities_df.empty:
                                n_y_unc_index = younger_unconformities_df.iloc[-1]['interface_id']
                                above_interfaces_for_horizon.append(n_y_unc_index)
                            break
                        else:
                            # erosional or basement
                            above_interfaces_for_horizon.append(younger_interface_index)
                            break
            else:
                younger_interface_ids = self.series_info_df[(self.series_info_df['interface_id'] < interface_idx)]['interface_id']
                if not younger_interface_ids.empty:
                    for younger_interface_idx in younger_interface_ids:
                        above_interfaces_for_horizon.append(younger_interface_idx)
            above_interfaces_for_horizon = list(set(above_interfaces_for_horizon))
            above_interfaces_for_horizon.sort()
            if not above_interfaces_for_horizon:
                above_interfaces_for_horizon = None
                above_series_id_for_horizon = None
            else:
                above_series_id_for_horizon = [self.series_info_df.iloc[idx]['series_id']
                                               for idx in above_interfaces_for_horizon]
            above_interfaces.append(above_interfaces_for_horizon)
            above_series.append(above_series_id_for_horizon)
        above_interfaces = above_interfaces[::-1]
        above_series = above_series[::-1]

        above_below_horizons_and_series = {}
        interface_unique_s_ids = []
        above_below_info = []
        for interface_idx in self.series_info_df.index:
            above_unique_series_id = np.unique(above_series[interface_idx])
            below_unique_series_id = np.unique(below_series[interface_idx])
            interface_idx_unique_s_ids = []
            if above_unique_series_id[0] is None:
                above_unique_series_id = None
            else:
                above_unique_series_id = above_unique_series_id.tolist()
                interface_idx_unique_s_ids.append(above_unique_series_id)
            if below_unique_series_id[0] is None:
                below_unique_series_id = None
            else:
                below_unique_series_id = below_unique_series_id.tolist()
                interface_idx_unique_s_ids.append(below_unique_series_id)
            if interface_idx_unique_s_ids:
                flat_list = [s_id for sub_unique_s_ids in interface_idx_unique_s_ids for s_id in sub_unique_s_ids]
                flat_list = list(set(flat_list))
                interface_unique_s_ids.append(flat_list)
                above_below_info.append([above_interfaces[interface_idx], below_interfaces[interface_idx],
                                         above_series[interface_idx], below_series[interface_idx]])
            else:
                interface_unique_s_ids.append(None)
                above_below_info.append([None, None, None, None])
            above_below_horizons_and_series[interface_idx] = {'above_horizons': above_interfaces[interface_idx],
                                                              'above_series': above_series[interface_idx],
                                                              'above_u_series': above_unique_series_id,
                                                              'below_horizons': below_interfaces[interface_idx],
                                                              'below_series': below_series[interface_idx],
                                                              'below_u_series': below_unique_series_id,
                                                              }
        self.unique_s_ids_for_interfaces_above_below = interface_unique_s_ids
        self.above_below_info = above_below_info
        above_below_interfaces_for_interfaces = pd.DataFrame.from_dict(above_below_horizons_and_series, orient='index')
        self.series_info_df = self.series_info_df.join(above_below_interfaces_for_interfaces, lsuffix='_caller',
                                                       rsuffix='_other')

        self.above_below_horizons_and_series_for_horizons = above_below_horizons_and_series
