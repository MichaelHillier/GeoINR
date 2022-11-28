import torch
import torch.nn as nn


def weighted_mse_interface_loss(pred, obs, weight, method='mean'):
    if method == 'mean':
        return torch.mean(weight * (pred - obs) ** 2)
    elif method == 'sum':
        return torch.sum(weight * (pred - obs) ** 2)
    else:
        return weight * (pred - obs) ** 2


def weighted_L1_interface_loss(pred, obs, weight, method='mean'):
    if method == 'mean':
        return torch.mean(weight * torch.abs(pred - obs))
    elif method == 'sum':
        return torch.sum(weight * torch.abs(pred - obs))
    else:
        return weight * torch.abs(pred - obs)


def weighted_L1_interface_loss_wnorm(pred, obs, weight, grad_pred, method='mean'):
    grad_norm_pred = torch.norm(grad_pred, p=2, dim=1)
    grad_error_term = 100 * torch.exp(-10 * grad_norm_pred) + 1
    # grad_error_term = -torch.log(grad_norm_pred)
    grad_error_term_np = grad_error_term.detach().cpu().numpy()
    grad_norm_np = grad_norm_pred.detach().cpu().numpy()
    norm_min = grad_norm_pred.min()
    norm_max = grad_norm_pred.max()
    norm_std, norm_mean = torch.std_mean(grad_norm_pred, unbiased=True)
    grad_pred_np = grad_pred.detach().cpu().numpy()
    grad_error_term = grad_error_term.squeeze()

    if method == 'mean':
        return torch.mean(weight * grad_error_term * torch.abs(pred - obs))
    elif method == 'sum':
        return torch.sum(weight * grad_error_term * torch.abs(pred - obs))
    else:
        return weight * grad_error_term * torch.abs(pred - obs)


def interface_loss(pred, obs, method='mean'):
    """
    Parameters
    ----------
    pred: Interface predictions outputted by NN
    obs: Constraints to compare predictons with.
    method: ['sum', 'mean'] how individual errors are combined

    Note: It is important that tensor's pred and obs must align. E.g. pred[i] corresponds to obs[i].
    AND pred.size() == obs.size()

    Returns
    -------
    scalar value representing all the individual computed errors between predictions and observations.
    """
    if method == 'mean':
        MSE_loss = nn.MSELoss()
    elif method == 'sum':
        MSE_loss = nn.MSELoss(reduction='sum')
    else:
        MSE_loss = nn.MSELoss(reduction='none')

    return MSE_loss(pred, obs)


def interface_l1_loss(pred, obs, method='mean'):
    """
    Parameters
    ----------
    pred: Interface predictions outputted by NN
    obs: Constraints to compare predictons with.
    method: ['sum', 'mean'] how individual errors are combined

    Note: It is important that tensor's pred and obs must align. E.g. pred[i] corresponds to obs[i].
    AND pred.size() == obs.size()

    Returns
    -------
    scalar value representing all the individual computed errors between predictions and observations.
    """

    if method == 'mean':
        L1 = nn.L1Loss()
        return L1(pred, obs)
    elif method == 'sum':
        L1 = nn.L1Loss(reduction='sum')
        return L1(pred, obs)
    else:
        L1 = nn.L1Loss(reduction='none')
        return L1(pred, obs)
