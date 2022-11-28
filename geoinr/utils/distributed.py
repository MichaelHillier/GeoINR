import numpy as np
from geoinr.input.constraints.interface import InterfaceData
from geoinr.input.constraints.unit import UnitData
from geoinr.input.grids import Grid


def combine_dist_interface_results(interface_dict: dict, interface_data: InterfaceData):
    n_pieces = len(interface_dict)
    scalar_pred = []
    residuals = []
    grad_norm_pred = []
    horizon_scalar_means = []
    horizon_residual_means = []
    horizon_variance = []
    original_indices = []
    for i in range(n_pieces):
        interface_data_i = interface_dict[i]
        original_indices.append(interface_data_i.original_interface_indices)
        if interface_data_i.scalar_pred is not None:
            scalar_pred.append(interface_data_i.scalar_pred)
        if interface_data_i.residuals is not None:
            residuals.append(interface_data_i.residuals)
        if interface_data_i.grad_norm_pred is not None:
            grad_norm_pred.append(interface_data_i.grad_norm_pred)
        if interface_data_i.horizon_scalar_means is not None:
            horizon_scalar_means.append(interface_data_i.horizon_scalar_means)
        if interface_data_i.horizon_residual_means is not None:
            horizon_residual_means.append(interface_data_i.horizon_residual_means)
        if interface_data_i.horizon_variance is not None:
            horizon_variance.append(interface_data_i.horizon_variance)
    n_interface = interface_data.n_interface
    original_indices = np.concatenate(original_indices)
    if scalar_pred:
        scalar_pred_combined = np.zeros(n_interface)
        scalar_pred = np.concatenate(scalar_pred)
        scalar_pred_combined[original_indices] = scalar_pred.flatten()
        interface_data.set_scalar_pred(scalar_pred_combined)
    if residuals:
        residuals_combined = np.zeros(n_interface)
        residuals = np.concatenate(residuals)
        residuals_combined[original_indices] = residuals.flatten()
        interface_data.set_residuals(residuals_combined)
    if grad_norm_pred:
        grad_norm_pred_combined = np.zeros(n_interface)
        grad_norm_pred = np.concatenate(grad_norm_pred)
        grad_norm_pred_combined[original_indices] = grad_norm_pred.flatten()
        interface_data.set_grad_norm_pred(grad_norm_pred_combined)
    if horizon_scalar_means:
        horizon_scalar_means = np.stack(horizon_scalar_means)
        horizon_scalar_means = np.mean(horizon_scalar_means, axis=0)
        interface_data.set_horizon_scalar_means(horizon_scalar_means)
    if horizon_residual_means:
        horizon_residual_means = np.stack(horizon_residual_means)
        horizon_residual_means = np.mean(horizon_residual_means, axis=0)
        interface_data.set_horizon_residual_means(horizon_residual_means)
    if horizon_variance:
        horizon_variance = np.stack(horizon_variance)
        horizon_variance = np.mean(horizon_variance, axis=0)
        interface_data.set_horizon_variance(horizon_variance)
    return interface_data


def combine_dist_unit_results(unit_dict: dict, unit_data: UnitData):
    n_pieces = len(unit_dict)
    unit_pred = []
    residuals = []
    class_residual_means = []
    horizon_decision_boundary_losses = []
    horizon_decision_boundary_probabilities = []
    original_indices = []
    for i in range(n_pieces):
        unit_data_i = unit_dict[i]
        original_indices.append(unit_data_i.original_unit_indices)
        if unit_data_i.unit_pred is not None:
            unit_pred.append(unit_data_i.unit_pred)
        if unit_data_i.residuals is not None:
            residuals.append(unit_data_i.residuals)
        if unit_data_i.class_residual_means is not None:
            class_residual_means_arr = np.fromiter(unit_data_i.class_residual_means.values(), dtype=float)
            class_residual_means.append(class_residual_means_arr)
        if unit_data_i.horizon_decision_boundary_losses is not None:
            horizon_decision_boundary_losses.append(unit_data_i.horizon_decision_boundary_losses)
        if unit_data_i.horizon_decision_boundary_probabilities is not None:
            horizon_decision_boundary_probabilities.append(unit_data_i.horizon_decision_boundary_probabilities)
    n_class_pts = unit_data.n_class_pts
    original_indices = np.concatenate(original_indices)
    if unit_pred:
        unit_pred_combined = np.zeros(n_class_pts)
        unit_pred = np.concatenate(unit_pred)
        unit_pred_combined[original_indices] = unit_pred.flatten()
        unit_data.set_unit_pred(unit_pred_combined)
    if residuals:
        residuals_combined = np.zeros(n_class_pts)
        residuals = np.concatenate(residuals)
        residuals_combined[original_indices] = residuals.flatten()
        unit_data.set_residuals(residuals_combined)
    if class_residual_means:
        class_residual_means = np.stack(class_residual_means)
        class_residual_means = np.mean(class_residual_means, axis=0)
        unit_data.set_class_residuals_means(class_residual_means)
    if horizon_decision_boundary_losses:
        horizon_decision_boundary_losses = np.stack(horizon_decision_boundary_losses)
        horizon_decision_boundary_losses = np.mean(horizon_decision_boundary_losses, axis=0)
        unit_data.set_horizon_decision_boundary_losses(horizon_decision_boundary_losses)
    if horizon_decision_boundary_probabilities:
        horizon_decision_boundary_probabilities = np.stack(horizon_decision_boundary_probabilities)
        horizon_decision_boundary_probabilities = np.mean(horizon_decision_boundary_probabilities, axis=0)
        unit_data.set_horizon_decision_boundary_probabilities(horizon_decision_boundary_probabilities)
    return unit_data


def combine_dist_grid_results(grid_dict: dict, grid_data: Grid):
    n_pieces = len(grid_dict)
    scalar_pred = []
    scalar_series = []
    scalar_grad_pred = []
    scalar_grad_norm_pred = []
    unit_pred = []
    for i in range(n_pieces):
        grid_data_i = grid_dict[i]
        if grid_data_i.scalar_pred is not None:
            scalar_pred.append(grid_data_i.scalar_pred)
        if grid_data_i.scalar_series is not None:
            scalar_series.append(grid_data_i.scalar_series)
        if grid_data_i.scalar_grad_pred is not None:
            scalar_grad_pred.append(grid_data_i.scalar_grad_pred)
        if grid_data_i.scalar_grad_norm_pred is not None:
            scalar_grad_norm_pred.append(grid_data_i.scalar_grad_norm_pred)
        if grid_data_i.unit_pred is not None:
            unit_pred.append(grid_data_i.unit_pred)
    if scalar_pred:
        scalar_pred = np.concatenate(scalar_pred)
        grid_data.set_scalar_pred(scalar_pred)
    if scalar_series:
        scalar_series = np.concatenate(scalar_series)
        grid_data.set_scalar_series(scalar_series)
    if scalar_grad_pred:
        scalar_grad_pred = np.concatenate(scalar_grad_pred)
        grid_data.set_scalar_grad_pred(scalar_grad_pred)
    if scalar_grad_norm_pred:
        scalar_grad_norm_pred = np.concatenate(scalar_grad_norm_pred)
        grid_data.set_scalar_grad_norm_pred(scalar_grad_norm_pred)
    if unit_pred:
        unit_pred = np.concatenate(unit_pred)
        grid_data.set_unit_pred(unit_pred)
    return grid_data


def combine_dist_model_results(model_dict: dict):
    n_pieces = len(model_dict)
    assert type(model_dict[0]) == dict, "model_dict is not a dict of dict as required"
    # first find all property names
    model_results = {prop: [] for prop in model_dict[0].keys()}
    for result_dict in model_dict.values():
        for prop, arr in result_dict.items():
            model_results[prop].append(arr)
    # combine the list of arr (could be either scalar, vector, matrix)
    for prop, arr_list in model_results.items():
        # make sure elements in arr_list are the same type and dimension (1: scalar/vector, 2: matrix)
        arr0 = arr_list[0]
        arr_type = type(arr0)
        for arr_i in arr_list:
            assert type(arr_i) == arr_type, "array elements are not all the same for this model property"
            if arr_type == np.ndarray:
                assert arr_i.ndim == arr0.ndim, "the dim of array elements are not all the same for this model property"
        if arr_type is not np.ndarray:
            # this is a list of python numbers (e.g. loss numbers)
            model_results[prop] = np.mean(np.array(arr_list))
        else:
            arr_combined = np.stack(arr_list)
            model_results[prop] = np.mean(arr_combined, axis=0)
    return model_results
