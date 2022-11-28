import torch
from geoinr.utils import derivatives


def grad_loss(grad_pred, normals, method='mean'):
    """
    Parameters
    ----------
    grad_pred: grad scalar field predictions @ orientation observations (e.g. normals) locations
    normals: 3D vectors describing normal orientation [N_orientation, 3]

    method: ['sum', 'mean'] how individual errors are combined'

    Returns
    -------
    Scalar value describing accumulated errors between N_orientation vectors and the network's estimated orientations
    at these locations.
    """

    # Compute norm of each vertex's scalar field gradient
    # grad_pred_np = grad_pred.detach().cpu().numpy()
    grad_norm_pred = torch.norm(grad_pred, p=2, dim=1)
    # grad_norm_var = torch.var(grad_norm_pred)
    # grad_norm_mean = grad_norm_pred.mean()
    # grad_norm_pred_np = grad_norm_pred.detach().cpu().numpy()
    # normals_np = normals.detach().cpu().numpy()

    grad_inner_product = torch.einsum('ij, ij->i', normals, grad_pred)
    # grad_inner_product_np = grad_inner_product.detach().cpu().numpy()
    cosine = grad_inner_product / grad_norm_pred
    # cosine_np = cosine.detach().cpu().numpy()
    if method == 'mean':
        return torch.mean(1 - cosine)
    else:  # sum
        return torch.sum(1 - cosine)


def curvature_loss(y, x):

    ''' hessian of y wrt x
    y: shape (N, 1)
    x: shape (N, 3)
    '''
    meta_batch_size, num_observations = y.shape[:2]
    grad_y0 = torch.ones_like(y[..., 0]).to(y.device)
    grad_y = torch.ones_like(y).to(y.device)
    h = torch.zeros(meta_batch_size, x.shape[-1], x.shape[-1]).to(y.device)

    dydx = torch.autograd.grad(y, [x], grad_outputs=grad_y, create_graph=True)[0]

    for i in range(x.shape[-1]):
        h[..., i, :] = torch.autograd.grad(dydx[..., i], x, grad_outputs=grad_y0,
                                           create_graph=True)[0][..., :]

    h = (h + h.transpose(1, 2)) / 2

    h2 = h.matrix_power(2)

    h2_trace = torch.einsum('bii->b', h2)

    h_loss = h2_trace.mean()

    return h_loss


def norm_loss(grad_pred):
    grad_norm = torch.norm(grad_pred, p=2, dim=2)
    eikonal_constraint = grad_norm - 1
    return torch.abs(eikonal_constraint)


def norm_loss_multiple_series(scalar_pred, scalar_coords, series_dict):
    norm = []
    for s_id, interface_ids in series_dict.items():
        # computer scalar grad for this field
        scalar_grad = derivatives.gradient(scalar_pred[:, s_id], scalar_coords)
        grad_norm = torch.norm(scalar_grad, p=2, dim=1)
        norm_constraint = torch.abs(grad_norm - 1).mean().view(-1)
        norm.append(norm_constraint)
    norm = torch.cat(norm)
    return norm