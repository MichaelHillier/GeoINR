import numpy as np
import torch
from itertools import dropwhile
from geoinr.input.constraints.interface import Series


def cut_series_by_unconformities(scalar_fields: torch.Tensor, series: Series):

    n_grid = scalar_fields.size()[0]
    resultant_scalar = torch.ones(n_grid).to(scalar_fields.device)
    processed = torch.zeros(n_grid, dtype=torch.bool).to(scalar_fields.device)
    units = torch.ones(n_grid).long()

    unconformity_series_ids = series.get_unconformity_series_ids()
    cutting_series_ids = series.series_cutting_sequence

    unit_id = 0
    for i in range(unconformity_series_ids.size):
        above_horizon = scalar_fields[:, unconformity_series_ids[i]] >= series.mean_scalar_values_for_series[
            series.unconformity_dict[unconformity_series_ids[i]]]
        above_horizon_and_unprocessed = above_horizon & ~processed
        current_scalar_field = scalar_fields[:, cutting_series_ids[i]]
        resultant_scalar[above_horizon_and_unprocessed] = current_scalar_field[above_horizon_and_unprocessed]
        if series.cutting_series_is_onlap[i]:
            # current scalar field being processed is an onlap series
            # get mean scalar values for each horizon in the onlap series
            onlap_horizon_indices = series.series_dict[cutting_series_ids[i]]
            for onlap_horizon_index in onlap_horizon_indices:
                onlap_horizon_scalar_value = series.mean_scalar_values_for_series[onlap_horizon_index]
                above_onlap_horizon = current_scalar_field >= onlap_horizon_scalar_value
                above_onlap_horizon_and_unconformity = above_onlap_horizon & above_horizon_and_unprocessed
                above_onlap_horizon_and_unconformity_and_unprocessed = above_onlap_horizon_and_unconformity & ~processed
                # above_onlap_horizon_and_unprocessed = above_onlap_horizon & ~processed
                units[above_onlap_horizon_and_unconformity_and_unprocessed] = unit_id
                processed[above_onlap_horizon_and_unconformity_and_unprocessed] = True
                unit_id += 1
            # process below last onlap horizon and above unconformity horizon
            below_onlap_horizons = current_scalar_field < series.mean_scalar_values_for_series[onlap_horizon_indices[-1]]
            below_onlap_horizons_and_above_unconformity = below_onlap_horizons & above_horizon
            below_onlap_horizons_and_above_unconformity_and_unprocessed = \
                below_onlap_horizons_and_above_unconformity & ~processed
            units[below_onlap_horizons_and_above_unconformity_and_unprocessed] = unit_id
            processed[below_onlap_horizons_and_above_unconformity_and_unprocessed] = True
            unit_id += 1
        else:
            # current scalar field being processed is an unconformity series
            units[above_horizon_and_unprocessed] = unit_id
            processed[above_horizon_and_unprocessed] = True
            unit_id += 1

    # process below last horizon/basement
    below_basement = scalar_fields[:, unconformity_series_ids[-1]] < series.mean_scalar_values_for_series[
            series.unconformity_dict[unconformity_series_ids[-1]]]
    below_basement_and_unprocessed = below_basement & ~processed
    current_scalar_field = scalar_fields[:, cutting_series_ids[-1]]
    resultant_scalar[below_basement_and_unprocessed] = current_scalar_field[below_basement_and_unprocessed]
    units[below_basement_and_unprocessed] = unit_id
    processed[below_basement_and_unprocessed] = True

    return resultant_scalar, scalar_fields, units






