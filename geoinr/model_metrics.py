import torch
import numpy as np
from geoinr.utils import derivatives
from geoinr.input.constraints import series
from geoinr.input.constraints.interface import InterfaceData
from geoinr.input.constraints.unit import UnitData


def get_horizon_metrics_using_scalar_constraints(scalar_pred, interface_coords, residuals, interface: InterfaceData):
    assert scalar_pred.numel() == residuals.numel(), "number of loss values are different from the" \
                                                     " number of scalar predictions"

    # calculate gradient of scalar field, and its norm (magnitude)
    scalar_grad = derivatives.gradient(scalar_pred, interface_coords)
    grad_norm = torch.norm(scalar_grad, p=2, dim=1)

    horizon_interface_indices = interface.horizon_interface_indices

    # calculate the scalar mean for each horizon sampled by interface constraints (InterfaceData)
    horizon_values = [scalar_pred[horizon_interface_indices[i]] for i in range(len(horizon_interface_indices))]
    horizon_scalar_mean = [horizon_values[i].mean() for i in range(len(horizon_values))]
    horizon_scalar_mean = torch.stack(horizon_scalar_mean)

    horizon_scalar_var = [horizon_values[i].var() for i in range(len(horizon_values))]
    horizon_scalar_var = torch.stack(horizon_scalar_var)

    # get per horizon residual means from residuals of each constraint and the scalar mean of the horizon
    # it's associated with
    horizon_residuals_values = [residuals[horizon_interface_indices[i]] for i in range(len(horizon_interface_indices))]
    horizon_residuals_mean = [horizon_residuals_values[i].mean() for i in range(len(horizon_interface_indices))]
    horizon_residuals_mean = torch.stack(horizon_residuals_mean)

    # attribute these metrics to the interface constraint data structure (as numpy arrays)
    interface.set_scalar_pred(scalar_pred.detach().cpu().numpy())
    interface.set_residuals(residuals.detach().cpu().numpy())
    interface.set_grad_norm_pred(grad_norm.detach().cpu().numpy())
    interface.set_horizon_scalar_means(horizon_scalar_mean.detach().cpu().numpy())
    interface.set_horizon_variance(horizon_scalar_var.detach().cpu().numpy())
    interface.set_horizon_residual_means(horizon_residuals_mean.detach().cpu().numpy())

    return interface


def get_horizon_metrics(scalar_pred, interface_coords, interface: InterfaceData):
    # calculate gradient of scalar field, and its norm (magnitude)
    scalar_grad = derivatives.gradient(scalar_pred, interface_coords)
    grad_norm = torch.norm(scalar_grad, p=2, dim=1)

    horizon_interface_indices = interface.horizon_interface_indices
    interface_horizon_indices = interface.interface_horizon_index

    # calculate the scalar mean for each horizon sampled by interface constraints (InterfaceData)
    horizon_values = [scalar_pred[horizon_interface_indices[i]] for i in range(len(horizon_interface_indices))]
    horizon_scalar_mean = [horizon_values[i].mean() for i in range(len(horizon_values))]
    horizon_scalar_mean = torch.stack(horizon_scalar_mean)

    horizon_scalar_var = [horizon_values[i].var() for i in range(len(horizon_values))]
    horizon_scalar_var = torch.stack(horizon_scalar_var)

    # calculate the residuals for each interface constraint point (determined by horizon scalar means and grad norm
    # at the constraint point
    residuals = torch.abs(horizon_scalar_mean[interface_horizon_indices] - scalar_pred.squeeze()) / (grad_norm + 1e-2)

    # get per horizon residual means from residuals of each constraint and the scalar mean of the horizon
    # it's associated with
    horizon_residuals_values = [residuals[horizon_interface_indices[i]] for i in range(len(horizon_interface_indices))]
    horizon_residuals_mean = [horizon_residuals_values[i].mean() for i in range(len(horizon_interface_indices))]
    horizon_residuals_mean = torch.stack(horizon_residuals_mean)

    # attribute these metrics to the interface constraint data structure (as numpy arrays)
    interface.set_scalar_pred(scalar_pred.detach().cpu().numpy())
    interface.set_residuals(residuals.detach().cpu().numpy())
    interface.set_grad_norm_pred(grad_norm.detach().cpu().numpy())
    interface.set_horizon_scalar_means(horizon_scalar_mean.detach().cpu().numpy())
    interface.set_horizon_variance(horizon_scalar_var.detach().cpu().numpy())
    interface.set_horizon_residual_means(horizon_residuals_mean.detach().cpu().numpy())

    return interface


def get_horizon_metrics_for_multiple_series(scalar_pred, scalar_coords, interface: InterfaceData, series: dict):
    # scalar = []
    # residuals = []
    # grad_norm = []
    scalar = torch.zeros(scalar_pred.size()[0], device=scalar_pred.device)
    residuals = torch.zeros(scalar_pred.size()[0], device=scalar_pred.device)
    grad_norm = torch.zeros(scalar_pred.size()[0], device=scalar_pred.device)

    horizon_scalar_mean = []
    horizon_scalar_var = []
    horizon_residual_mean = []
    horizon_residual_std = []

    horizon_interface_indices = interface.horizon_interface_indices
    interface_horizon_indices = interface.interface_horizon_index
    for s_id, interface_ids in series.items():
        # computer scalar grad for this field
        scalar_grad = derivatives.gradient(scalar_pred[:, s_id], scalar_coords)
        grad_norm_series = torch.norm(scalar_grad, p=2, dim=1)

        # calculate the scalar mean for each horizon sampled by interface constraints (InterfaceData)
        # first get the appropriate series (s_id), and the indices of interface constraints for each
        # horizon in this series
        scalar_field_series = scalar_pred[:, s_id]
        series_horizon_interface_indices = [horizon_interface_indices[horizon_index] for horizon_index in interface_ids]
        n_interface_per_horizon = [horizon_i_interface_indices.numel()
                                   for horizon_i_interface_indices in series_horizon_interface_indices]
        n_horizons = len(n_interface_per_horizon)
        series_interface_horizon_indices = torch.arange(n_horizons).repeat_interleave(torch.tensor(n_interface_per_horizon))

        # calculate the scalar mean for each horizon in this series
        horizon_values = [scalar_field_series[series_horizon_interface_indices[i]]
                          for i in range(len(series_horizon_interface_indices))]
        horizon_scalar_mean_series = [horizon_values[i].mean() for i in range(len(horizon_values))]
        horizon_scalar_mean_series = torch.stack(horizon_scalar_mean_series)
        horizon_scalar_mean.append(horizon_scalar_mean_series)

        # calculate the variance in scalar values for each horizon in this series
        horizon_scalar_var_series = [horizon_values[i].var() for i in range(len(horizon_values))]
        horizon_scalar_var_series = torch.stack(horizon_scalar_var_series)
        horizon_scalar_var.append(horizon_scalar_var_series)

        # interface constraints indices for this series
        series_interface_indices = torch.cat(series_horizon_interface_indices)
        # series_interface_horizon_indices = interface_horizon_indices[series_interface_indices]
        scalar_field_at_series_interface_constraints = scalar_field_series[series_interface_indices]
        grad_norm_at_series_interface_constraints = grad_norm_series[series_interface_indices]
        # scalar.append(scalar_field_at_series_interface_constraints)
        # grad_norm.append(grad_norm_at_series_interface_constraints)
        scalar[series_interface_indices] = scalar_field_at_series_interface_constraints
        grad_norm[series_interface_indices] = grad_norm_at_series_interface_constraints

        # calculate the residuals for each interface constraint point (determined by horizon scalar means and grad norm
        # at the constraint point
        series_residual = torch.abs(horizon_scalar_mean_series[series_interface_horizon_indices] -
                                    scalar_field_at_series_interface_constraints.squeeze()) \
                          / (grad_norm_at_series_interface_constraints)
        #residuals.append(series_residual)
        residuals[series_interface_indices] = series_residual

        # get per horizon residual means from residuals of each constraint and the scalar mean of the horizon
        # it's associated with
        series_horizon_interface_reindexed_indices = torch.arange(series_interface_indices.numel()).split(n_interface_per_horizon)
        horizon_residual_values_series = [series_residual[series_horizon_interface_reindexed_indices[i]]
                                          for i in range(len(series_horizon_interface_reindexed_indices))]
        horizon_residual_mean_series = [horizon_residual_values_series[i].mean()
                                        for i in range(len(series_horizon_interface_indices))]
        horizon_residual_mean_series = torch.stack(horizon_residual_mean_series)
        horizon_residual_mean.append(horizon_residual_mean_series)
        horizon_residual_std_series = [horizon_residual_values_series[i].std(unbiased=False)
                                       for i in range(len(series_horizon_interface_indices))]
        horizon_residual_std_series = torch.stack(horizon_residual_std_series)
        horizon_residual_std.append(horizon_residual_std_series)

    # scalar = torch.cat(scalar)
    # residuals = torch.cat(residuals)
    # grad_norm = torch.cat(grad_norm)
    horizon_scalar_mean = torch.cat(horizon_scalar_mean)
    horizon_scalar_var = torch.cat(horizon_scalar_var)
    horizon_residual_mean = torch.cat(horizon_residual_mean)
    horizon_residual_std = torch.cat(horizon_residual_std)

    # attribute these metrics to the interface constraint data structure (as numpy arrays)
    interface.set_scalar_pred(scalar.detach().cpu().numpy())
    interface.set_residuals(residuals.detach().cpu().numpy())
    interface.set_grad_norm_pred(grad_norm.detach().cpu().numpy())
    interface.set_horizon_scalar_means(horizon_scalar_mean.detach().cpu().numpy())
    interface.set_horizon_variance(horizon_scalar_var.detach().cpu().numpy())
    interface.set_horizon_residual_means(horizon_residual_mean.detach().cpu().numpy())
    interface.set_horizon_residual_std(horizon_residual_std.detach().cpu().numpy())

    return interface


def get_unit_metrics(unit_pred, unit_data, units: UnitData):
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
    constraint_unit_losses = (ce_loss(unit_pred, unit_data)).detach().cpu().numpy()
    unit_losses_dict = {i: [] for i in range(units.n_classes)}
    unit_data = unit_data.detach().cpu().numpy()
    for i, unit_label in enumerate(unit_data):
        unit_losses_dict[unit_label].append(constraint_unit_losses[i])
    for i in range(units.n_classes):
        if len(unit_losses_dict[i]) > 0:
            unit_i_losses = np.array(unit_losses_dict[i])
            unit_losses_dict[i] = np.mean(unit_i_losses)
        else:
            unit_losses_dict[i] = np.array(0)

    # convert unit_pred (vector of probabilities) to class via argmax
    unit_pred = torch.argmax(unit_pred, dim=1)
    units.set_unit_pred(unit_pred.detach().cpu().numpy())
    units.set_residuals(constraint_unit_losses)
    units.set_class_residuals_means(unit_losses_dict)
    return units


def implicit_unit_error(s_above, s_grad_norm_above, horizon_s_above,
                        s_below, s_grad_norm_below, horizon_s_below):
    """
    Input into this function are the associated scalar field values for a set of unit
    constraints with the SAME unit id. Must be the same unit id. Otherwise, computations
    are meaningless.
    A unit is bounded from above and below by specific horizons. Each horizon is associated
    with a different scalar field.
    :param s_above: scalar values for set a unit points (same id) for all scalar fields above this unit
                    [n_unit_i_pts, n_horizons_above]
    :param s_grad_norm_above: scalar grad norms for set of unit points (same id) for all scalar fields
                              above this unit [n_unit_i_pts, n_horizons_above]
    :param horizon_s_above: mean scalar value for all horizons above this unit [1, n_horizons_above]
    :param s_below: scalar values for set a unit points (same id) for all scalar fields below this unit
                    [n_unit_i_pts, n_horizons_below]
    :param s_grad_norm_below: scalar grad norms for set of unit points (same id) for all scalar fields
                              below this unit [n_unit_i_pts, n_horizons_below]
    :param horizon_s_below: mean scalar value for all horizons below this unit [1, n_horizons_below]
    :return: mean error for unit i (scalar). For each point, all the errors wrt each series
    """

    if s_above is not None and s_below is not None:
        above = (s_above - horizon_s_above) / s_grad_norm_above
        above_error = torch.maximum(above, torch.tensor(0, device=above.device)).sum(dim=1)
        below = (s_below - horizon_s_below) / s_grad_norm_below
        below_error = torch.abs(torch.minimum(below, torch.tensor(0, device=below.device))).sum(dim=1)
        error = above_error + below_error
        return error
    else:
        if s_above is not None:
            above = (s_above - horizon_s_above) / s_grad_norm_above
            above_error = torch.maximum(above, torch.tensor(0, device=above.device)).sum(dim=1)
            return above_error
        elif s_below is not None:
            below = (s_below - horizon_s_below) / s_grad_norm_below
            below_error = torch.abs(torch.minimum(below, torch.tensor(0, device=below.device))).sum(dim=1)
            return below_error


def get_implicit_unit_metrics(scalar_pred, scalar_coords, unit_indices, series_struct: series.Series, units: UnitData):
    # generate grad norm
    scalar_grad = derivatives.jacobian(scalar_pred, scalar_coords)  # [n_pts, n_series, 3]
    scalar_grad_norm = torch.norm(scalar_grad, p=2, dim=2)

    constraint_unit_losses = np.zeros(scalar_pred.shape[0])
    unit_losses_dict = {}
    for unit_id in range(series_struct.n_unit_classes):
        unit_id_indices = unit_indices[unit_id]
        horizon_ids_above = series_struct.above_below_horizons_and_series_for_units[unit_id]['above_horizons']
        series_ids_above = series_struct.above_below_horizons_and_series_for_units[unit_id]['above_series']
        horizons_ids_below = series_struct.above_below_horizons_and_series_for_units[unit_id]['below_horizons']
        series_ids_below = series_struct.above_below_horizons_and_series_for_units[unit_id]['below_series']
        if len(unit_id_indices) == 0:
            unit_losses_dict[unit_id] = np.array(0)
            continue
        if horizon_ids_above is None:
            s_above = None
            s_grad_norm_above = None
            horizon_s_above = None
        else:
            s_above = scalar_pred[unit_id_indices][:, series_ids_above]
            s_grad_norm_above = scalar_grad_norm[unit_id_indices][:, series_ids_above]
            horizon_s_above = series_struct.mean_scalar_values_for_series[horizon_ids_above]
            horizon_s_above = torch.from_numpy(horizon_s_above).float().to(scalar_pred.device).view(1, -1)
            # horizon_s_above = series.mean_scalar_values_for_series.view(1, -1)[0, [horizon_ids_above]]
        if horizons_ids_below is None:
            s_below = None
            s_grad_norm_below = None
            horizon_s_below = None
        else:
            s_below = scalar_pred[unit_id_indices][:, series_ids_below]
            s_grad_norm_below = scalar_grad_norm[unit_id_indices][:, series_ids_below]
            horizon_s_below = series_struct.mean_scalar_values_for_series[horizons_ids_below]
            horizon_s_below = torch.from_numpy(horizon_s_below).float().to(scalar_pred.device).view(1, -1)
            # horizon_s_below = series.mean_scalar_values_for_series.view(1, -1)[0, [horizons_ids_below]]
        unit_id_error = implicit_unit_error(s_above, s_grad_norm_above, horizon_s_above,
                                            s_below, s_grad_norm_below, horizon_s_below)
        if isinstance(unit_id_indices, torch.Tensor):
            unit_id_indices = unit_id_indices.detach().cpu().numpy()
        constraint_unit_losses[unit_id_indices] = unit_id_error.detach().cpu().numpy().flatten()
        unit_losses_dict[unit_id] = unit_id_error.mean().detach().cpu().numpy().flatten()

    units.set_residuals(constraint_unit_losses)
    units.set_class_residuals_means(unit_losses_dict)
    return units
