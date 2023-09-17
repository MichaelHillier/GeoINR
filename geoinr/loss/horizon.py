import torch
from geoinr.input.constraints import series
from geoinr.utils import derivatives


def horizon_loss(scalar_pred, horizon_interface_indices, scalar_grad_pred):
    grad_norm_pred = torch.norm(scalar_grad_pred, p=2, dim=1)
    horizon_scalar_values = [scalar_pred[horizon_interface_indices[i]]
                             for i in range(len(horizon_interface_indices))]
    horizon_scalar_norms = [grad_norm_pred[horizon_interface_indices[i]]
                            for i in range(len(horizon_interface_indices))]

    horizon_scalar_mean = [horizon_scalar_values[i].mean() for i in range(len(horizon_scalar_values))]
    horizon_scalar_mean = torch.stack(horizon_scalar_mean)

    horizon_scalar_var = [torch.var(horizon_scalar_values[i]) for i in range(len(horizon_scalar_values))]
    horizon_scalar_var = torch.stack(horizon_scalar_var)

    horizon_dist_residuals = [
        torch.abs(horizon_scalar_mean[i] - horizon_scalar_values[i]) / (horizon_scalar_norms[i] + 1e-6)
        for i in range(len(horizon_interface_indices))]
    horizon_dist_mean_residuals = torch.stack([torch.mean(horizon_dist_residuals[i])
                                               for i in range(len(horizon_interface_indices))])

    return horizon_dist_mean_residuals, horizon_scalar_var, horizon_scalar_mean


def horizon_loss_multiple_series(scalar_pred, scalar_coords, horizon_interface_indices, series_dict: dict):
    residuals = []
    variance = []
    polarity = []
    mean = []
    # norm = []
    for s_id, interface_ids in series_dict.items():
        # computer scalar grad for this field
        scalar_grad = derivatives.gradient(scalar_pred[:, s_id], scalar_coords)
        # grad_norm = torch.norm(scalar_grad, p=2, dim=1)
        # norm_constraint = torch.abs(grad_norm - 1).mean().view(-1)

        # all the gradients should be pointing up (+z) to younger geology. If it isn't we need to penalize via a loss
        # note this simple approach is dangerous to use in general!! For sedimentary basins it's perfectly fine to use
        scalar_grad_z = scalar_grad[:, 2]
        polarity_loss = scalar_grad_z.minimum(torch.tensor(0)).abs()
        series_horizon_interface_indices = [horizon_interface_indices[horizon_index] for horizon_index in interface_ids]
        series_horizon_polarity_loss = torch.stack([polarity_loss[series_horizon_i_interface_indices].mean()
                                                    for series_horizon_i_interface_indices in
                                                    series_horizon_interface_indices])
        res, var, avg = horizon_loss(scalar_pred[:, s_id], series_horizon_interface_indices, scalar_grad)
        residuals.append(res)
        variance.append(var)
        mean.append(avg)
        #norm.append(norm_constraint)
        polarity.append(series_horizon_polarity_loss)

    residuals = torch.cat(residuals)
    variance = torch.cat(variance)
    #norm = torch.cat(norm)
    mean = torch.cat(mean)
    polarity = torch.cat(polarity)

    return residuals, variance, polarity, mean


def stratigraphic_above_below_error(s_above, s_grad_norm_above, horizon_s_above,
                                    s_below, s_grad_norm_below, horizon_s_below):
    """
    Input into this function are the associated scalar field values for a set of point
    constraints with the SAME horizon/interface id. Must be the same horizon/interface id I_k. Otherwise, computations
    are meaningless.
    Each horizon is associated with a different scalar field.
    :param s_above: scalar values for set of interface points from I_k for all scalar fields above this interface I_k
                    [n_interface_pts, n_horizons_above]
    :param s_grad_norm_above: scalar grad norms for set of interface points from I_k for all scalar fields above
                              this interface I_k [n_interface_pts, n_horizons_above]
    :param horizon_s_above: mean scalar value for all horizons above this unit [1, n_horizons_above]
    :param s_below: scalar values for set of interface points from I_k for all scalar fields below this interface I_k
                    [n_interface_pts, n_horizons_above]
    :param s_grad_norm_below: scalar grad norms for set of interface points from I_k for all scalar fields below
                              this interface I_k [n_interface_pts, n_horizons_above]
    :param horizon_s_below: mean scalar value for all horizons below this unit [1, n_horizons_below]
    :return: mean error for points sampling horizon/interface I_k (scalar).
    """

    if s_above is not None and s_below is not None:
        strat_rel_above = (s_above - horizon_s_above) / s_grad_norm_above
        strat_rel_above_error = torch.maximum(strat_rel_above, torch.tensor(0, device=strat_rel_above.device)).sum(dim=1)
        strat_rel_below = (s_below - horizon_s_below) / s_grad_norm_below
        strat_rel_below_error = torch.abs(torch.minimum(strat_rel_below, torch.tensor(0, device=strat_rel_below.device))).sum(dim=1)
        error = strat_rel_above_error + strat_rel_below_error
        # strat_rel_above_np = strat_rel_above.detach().cpu().numpy()
        # strat_rel_above_error_np = strat_rel_above_error.detach().cpu().numpy()
        # strat_rel_below_np = strat_rel_below.detach().cpu().numpy()
        # strat_rel_below_error_np = strat_rel_below_error.detach().cpu().numpy()
        return error.mean()
    else:
        if s_above is not None:
            # Above (younger) scalar fields evaluated using a point set sampling an interface I_k below (older)
            # the following stratigraphic relation should be -ve (these points should be BELOW the Above interface)
            # if relation is +ve (ABOVE) than non-zero error
            # if relation is -ve (BELOW) than zero error (respects constraint)
            strat_rel = (s_above - horizon_s_above) / s_grad_norm_above
            strat_rel_error = torch.maximum(strat_rel, torch.tensor(0, device=strat_rel.device)).sum(dim=1)
            # strat_rel_np = strat_rel.detach().cpu().numpy()
            # strat_rel_error_np = strat_rel_error.detach().cpu().numpy()
            return strat_rel_error.mean()
        elif s_below is not None:
            # Below (older) scalar fields evaluated using a point set sampling an interface I_k above (younger)
            # the following stratigraphic relation should be +ve (these points should be ABOVE the Below interface)
            # if relation is +ve (ABOVE) than zero error (respects constraint)
            # if relation is -ve (BELOW) than non-zero error
            strat_rel = (s_below - horizon_s_below) / s_grad_norm_below
            strat_rel_error = torch.abs(torch.minimum(strat_rel, torch.tensor(0, device=strat_rel.device))).sum(dim=1)
            # strat_rel_np = strat_rel.detach().cpu().numpy()
            # strat_rel_error_np = strat_rel_error.detach().cpu().numpy()
            return strat_rel_error.mean()


def stratigraphic_above_below_losses(scalar_pred, scalar_coords, horizon_interface_indices, series_struct: series.Series):
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

    horizon_i_losses = []
    for horizon_id in range(series_struct.n_horizons):
        horizon_id_point_indices = horizon_interface_indices[horizon_id]
        horizon_ids_above = series_struct.above_below_horizons_and_series_for_horizons[horizon_id]['above_horizons']
        series_ids_above = series_struct.above_below_horizons_and_series_for_horizons[horizon_id]['above_series']
        horizons_ids_below = series_struct.above_below_horizons_and_series_for_horizons[horizon_id]['below_horizons']
        series_ids_below = series_struct.above_below_horizons_and_series_for_horizons[horizon_id]['below_series']
        have_horizons = False
        if horizon_ids_above is None:
            s_above = None
            s_grad_norm_above = None
            horizon_s_above = None
        else:
            have_horizons = True
            s_above = scalar_pred[horizon_id_point_indices][:, series_ids_above]
            s_grad_norm_above = scalar_grad_norm[horizon_id_point_indices][:, series_ids_above]
            horizon_s_above = series_struct.mean_scalar_values_for_series.view(1, -1)[0, [horizon_ids_above]]
            # s_above_np = s_above.detach().cpu().numpy()
            # s_grad_norm_above_np = s_grad_norm_above.detach().cpu().numpy()
        if horizons_ids_below is None:
            s_below = None
            s_grad_norm_below = None
            horizon_s_below = None
        else:
            have_horizons = True
            s_below = scalar_pred[horizon_id_point_indices][:, series_ids_below]
            s_grad_norm_below = scalar_grad_norm[horizon_id_point_indices][:, series_ids_below]
            horizon_s_below = series_struct.mean_scalar_values_for_series.view(1, -1)[0, [horizons_ids_below]]
            # s_below_np = s_below.detach().cpu().numpy()
            # s_grad_norm_below_np = s_grad_norm_below.detach().cpu().numpy()
        if have_horizons:
            horizon_id_error = stratigraphic_above_below_error(s_above, s_grad_norm_above, horizon_s_above,
                                                               s_below, s_grad_norm_below, horizon_s_below)
            horizon_i_losses.append(horizon_id_error)

    horizon_losses_all = torch.stack(horizon_i_losses)
    return horizon_losses_all
