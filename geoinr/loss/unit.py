import torch
import torch.nn as nn
from geoinr.input.constraints import series
from geoinr.utils import derivatives


def unit_classification_loss(unit_pred, unit_labels):
    loss_fcn = nn.CrossEntropyLoss()
    return loss_fcn(unit_pred, unit_labels)


def class_loss(class_pred, labels):
    loss_fcn = nn.CrossEntropyLoss()
    return loss_fcn(class_pred, labels)


def unit_error(s_above, s_grad_norm_above, horizon_s_above,
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
        # debug
        # above_np = above.detach().cpu().numpy()
        # above_error_np = above_error.detach().cpu().numpy()
        # below_np = below.detach().cpu().numpy()
        # below_error_np = below_error.detach().cpu().numpy()
        # non_zero = error.count_nonzero()
        # return error.sum() / non_zero
        return error.mean()
    else:
        if s_above is not None:
            above = (s_above - horizon_s_above) / s_grad_norm_above
            above_error = torch.maximum(above, torch.tensor(0, device=above.device)).sum(dim=1)
            # above_np = above.detach().cpu().numpy()
            # above_error_np = above_error.detach().cpu().numpy()
            # non_zero = above_error.count_nonzero()
            #return above_error.sum() / non_zero
            return above_error.mean()
        elif s_below is not None:
            below = (s_below - horizon_s_below) / s_grad_norm_below
            below_error = torch.abs(torch.minimum(below, torch.tensor(0, device=below.device))).sum(dim=1)
            # below_np = below.detach().cpu().numpy()
            # below_error_np = below_error.detach().cpu().numpy()
            # non_zero = below_error.count_non_zero()
            # return below_error.sum() / non_zero
            return below_error.mean()


def unit_losses(scalar_pred, scalar_coords, unit_indices, series_struct: series.Series):
    # generate grad norm
    scalar_grad = derivatives.jacobian(scalar_pred, scalar_coords)  # [n_pts, n_series, 3]
    scalar_grad_norm = torch.norm(scalar_grad, p=2, dim=2)
    # scalar_grad_np = scalar_grad.detach().cpu().numpy()

    # debug
    # scalar_grad0 = derivatives.gradient(scalar_pred[:, 0], scalar_coords)
    # scalar_grad1 = derivatives.gradient(scalar_pred[:, 1], scalar_coords)
    # scalar_grad0_np = scalar_grad0.detach().cpu().numpy()
    # scalar_grad1_np = scalar_grad1.detach().cpu().numpy()
    # scalar_pred_np = scalar_pred.detach().cpu().numpy()

    unit_i_losses = []
    for unit_id in range(series_struct.n_unit_classes):
        unit_id_indices = unit_indices[unit_id]
        horizon_ids_above = series_struct.above_below_horizons_and_series_for_units[unit_id]['above_horizons']
        series_ids_above = series_struct.above_below_horizons_and_series_for_units[unit_id]['above_series']
        horizons_ids_below = series_struct.above_below_horizons_and_series_for_units[unit_id]['below_horizons']
        series_ids_below = series_struct.above_below_horizons_and_series_for_units[unit_id]['below_series']
        if len(unit_id_indices) == 0:
            continue
        if horizon_ids_above is None:
            s_above = None
            s_grad_norm_above = None
            horizon_s_above = None
        else:
            s_above = scalar_pred[unit_id_indices][:, series_ids_above]
            s_grad_norm_above = scalar_grad_norm[unit_id_indices][:, series_ids_above]
            horizon_s_above = series_struct.mean_scalar_values_for_series.view(1, -1)[0, [horizon_ids_above]]
            # s_above_np = s_above.detach().cpu().numpy()
            # s_grad_norm_above_np = s_grad_norm_above.detach().cpu().numpy()
        if horizons_ids_below is None:
            s_below = None
            s_grad_norm_below = None
            horizon_s_below = None
        else:
            s_below = scalar_pred[unit_id_indices][:, series_ids_below]
            s_grad_norm_below = scalar_grad_norm[unit_id_indices][:, series_ids_below]
            horizon_s_below = series_struct.mean_scalar_values_for_series.view(1, -1)[0, [horizons_ids_below]]
            # s_below_np = s_below.detach().cpu().numpy()
            # s_grad_norm_below_np = s_grad_norm_below.detach().cpu().numpy()
        unit_id_error = unit_error(s_above, s_grad_norm_above, horizon_s_above,
                                   s_below, s_grad_norm_below, horizon_s_below)
        unit_i_losses.append(unit_id_error)

    unit_losses_all = torch.stack(unit_i_losses)
    return unit_losses_all

