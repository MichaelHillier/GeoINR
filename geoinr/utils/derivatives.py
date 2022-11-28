import torch


def gradient(y, x, grad_outputs=None):
    """
    Compute the derivative of y wrt x, where x can be a vector. e.g. x = [x, y, z] - this will compute
    the spatial gradient of y at position x.
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def divergence(y, x):
    """
    Compute the divergence of y wrt x.
    """
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def laplace(y, x):
    """
    Compute the laplacian of y wrt x. E.g. The divergence of the gradient.
    """
    grad = gradient(y, x)
    return divergence(grad, x)


def hessian(y, x):
    """ hessian of y wrt x
    y: shape (N, 1)
    x: shape (N, 3)
    """
    meta_batch_size, num_observations = y.shape[:2]
    grad_y0 = torch.ones_like(y[..., 0]).to(y.device)
    grad_y = torch.ones_like(y).to(y.device)
    h = torch.zeros(meta_batch_size, x.shape[-1], x.shape[-1]).to(y.device)

    dydx = torch.autograd.grad(y, [x], grad_outputs=grad_y, create_graph=True, retain_graph=True)[0]

    for i in range(x.shape[-1]):
        h[..., i, :] = torch.autograd.grad(dydx[..., i], x, grad_outputs=grad_y0,
                                           create_graph=True, retain_graph=True)[0][..., :]

    # h = (h + h.transpose(1, 2)) / 2
    # Above done for numerical reasons. In some situations the hessian
    # is slightly off symmetric. This was symmetricalize the hessian

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h


def hessian_series(y, x):
    """ hessian of y wrt x
    y: shape (N, 1)
    x: shape (N, 3)
    """
    meta_batch_size, num_observations = y.shape[:2]
    grad_y0 = torch.ones_like(y)
    grad_y = torch.ones_like(y[:, 0])
    h = torch.zeros(meta_batch_size, y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)

    dydx = []
    for i in range(y.shape[-1]):
        dydx.append(torch.autograd.grad(y[..., i], x, grad_outputs=grad_y, create_graph=True, retain_graph=True)[0])
    dydx = torch.cat(dydx, dim=1)
    dydx = dydx.view(-1, y.shape[-1], x.shape[-1])

    for i in range(y.shape[-1]):
        for j in range(x.shape[-1]):
            d1 = dydx[:, i, j]
            ff = torch.autograd.grad(dydx[:, i, j], x, grad_outputs=grad_y,
                                     create_graph=True, retain_graph=True)[0][..., :]
            h[:, i, j, :] = ff
            t = 6
            #h[:, i, :, j] = ff

    h = h ** 2
    h = h.view(y.shape[0], y.shape[-1], -1).sum(2).mean()

    # for i in range(x.shape[-1]):
    #     ff = torch.autograd.grad(dydx[..., i], x, grad_outputs=grad_y0,
    #                              create_graph=True, retain_graph=True)[0][..., :]
    #     t = 6
    #     # h[..., i, :] = torch.autograd.grad(dydx[..., i], x, grad_outputs=grad_y0,
    #     #                                    create_graph=True, retain_graph=True)[0][..., :]
    return h


def jacobian(y, x):
    jacobi = []
    grad_outputs = torch.ones_like(y[:, 0])

    for i in range(y.shape[-1]):
        jacobi.append(torch.autograd.grad(y[..., i], x, grad_outputs=grad_outputs, create_graph=True)[0])
    jacobi = torch.cat(jacobi, dim=1)

    return jacobi.view(-1, y.shape[-1], x.shape[-1])


def curvature(y, x):
    """ Computes the curvature of y wrt x. x must be 2/3 D.
    y: shape (N, 1)
    x: shape (N, 3)
    """
    meta_batch_size, num_observations = y.shape[:2]
    grad_y0 = torch.ones_like(y[..., 0]).to(y.device)
    grad_y = torch.ones_like(y).to(y.device)
    h = torch.zeros(meta_batch_size, x.shape[-1], x.shape[-1]).to(y.device)

    dydx = torch.autograd.grad(y, [x], grad_outputs=grad_y, create_graph=True)[0]

    for i in range(x.shape[-1]):
        h[..., i, :] = torch.autograd.grad(dydx[..., i], x, grad_outputs=grad_y0,
                                           create_graph=True)[0][..., :]

    h = (h + h.transpose(1, 2)) / 2

    h_trace = torch.einsum('bii->b', h)
    grad_norm = torch.norm(dydx, p=2, dim=1)

    curvature_t1 = torch.matmul(dydx.view(-1, 1, 3), h)
    curvature_t1 = torch.matmul(curvature_t1, dydx.view(-1, 3, 1)).flatten()

    curvature_t2 = (grad_norm ** 2) * h_trace

    mean_curvature = (curvature_t1 - curvature_t2) / (2 * (grad_norm ** 3))

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return mean_curvature
