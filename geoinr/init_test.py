import numpy as np
import torch
from geoinr.model.mlp.model import SimpleMLP
from geoinr.input.grids import generate_vtk_structure_grid_and_grid_points
import geoinr.args as args
from matplotlib import pyplot as plt, colors
from mpl_toolkits.mplot3d import Axes3D

prog_args = args.arg_parse()

bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

grid, sgrid = generate_vtk_structure_grid_and_grid_points(bounds, 0.1, 0.1)
dims = sgrid.GetDimensions()

model = SimpleMLP(3, prog_args.embed_dim, prog_args.num_hidden_layers, 1, prog_args.concat)

# for name, weight in model.named_parameters():
#     if 'weight' in name:
#         plt.figure()
#         plt.imshow(weight.detach().cpu().numpy(), interpolation='none', aspect='auto')
#         t = 5

gridT = torch.from_numpy(grid).float()
grid_scalar, _ = model(gridT)

grid_scalar = grid_scalar.detach().numpy()
grid_scalar_3D = grid_scalar.reshape(dims[2], dims[0], dims[1])

filled = np.ones((dims[2], dims[0], dims[1]), dtype=np.bool)

volume = np.random.rand(dims[2], dims[0], dims[1])
color_values = np.repeat(volume[:, :, :, np.newaxis], 3, axis=3)

cmap = plt.cm.gist_rainbow
norm = colors.Normalize(vmin=grid_scalar.min(), vmax=grid_scalar.max())
grid_scalar_colors = cmap(norm(grid_scalar))
grid_scalar_colors_3D = grid_scalar_colors.reshape(dims[2], dims[0], dims[1], 4)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(filled, facecolors=grid_scalar_colors_3D)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.axis('off')
plt.show()
t = 5
