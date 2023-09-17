import os
import socket
from datetime import datetime
import torch
import time
from torch.utils.tensorboard import SummaryWriter

import geoinr.args as args
from geoinr.input.scalers import get_scaler
from geoinr.input.constraints.interface import InterfaceData
from geoinr.utils.vtk_utils import get_resultant_bounds_from_vtk_objects
from geoinr.input.grids import Grid, GridPointDataDistributedSampler, GridData, \
    generate_vtk_structure_grid_and_grid_points
from geoinr.utils.distributed import combine_dist_grid_results
from geoinr.model.mlp.model import SimpleMLP
from geoinr.output import ModelOutput
import geoinr.loss.interface as iloss
import geoinr.model_metrics as mm
from geoinr.utils import derivatives

prog_args = args.arg_parse()
options = vars(prog_args)

# Load Data
# Get Interface Data
interface_file = os.path.join(args.ROOT_DIR, prog_args.datadir, prog_args.dataset, prog_args.interface_file)
interface = InterfaceData(interface_file, 1)

# get data bounds from combined datasets; interface, orientation, classes. Used for feature normalization/scaling
bounds = get_resultant_bounds_from_vtk_objects(interface.interface_vtk,
                                               xy_buffer=prog_args.xy_buffer,
                                               z_buffer=prog_args.z_buffer)
# get normalization/scaling function
scalar = get_scaler(prog_args.scale_method, interface.coords, prog_args.scale_range)

# normalize/scale features
interface.transform(scalar)
interface.convert_to_torch()
interface.send_to_gpu(0)

# Get Grid Point Data
# Create grid points first
grid_coords, grid = generate_vtk_structure_grid_and_grid_points(bounds, 1.5, 1.5)
grid_points = Grid(grid_coords, grid, interface.series)
# normalize/scale grid features
grid_points.transform(scalar)
# split/chunk evaluation/inference data for distributed computing
grid_dist = GridPointDataDistributedSampler(grid_points, 8)
grid_points.uniformly_sample_grid_for_points(prog_args.n_grid_samples)
grid_points.transform_sampled_grid_points(scalar)
sampled_grid_points = torch.from_numpy(grid_points.sampled_grid_points).float().to(0)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join(args.ROOT_DIR, 'runs', current_time + '_' + socket.gethostname())
writer = SummaryWriter(log_dir=log_dir)
prog_args.model_dir = writer.log_dir

# Create INR Model
model = SimpleMLP(3, prog_args, 1).to(0)

optimizer = torch.optim.AdamW(model.parameters(), lr=prog_args.learning_rate, weight_decay=prog_args.weight_decay,
                              amsgrad=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=prog_args.num_epocs, eta_min=0,
                                                       last_epoch=-1, verbose=False)

dur = []

for epoch in range(prog_args.num_epocs):
    model.train()
    t0 = time.time()
    # forward
    interface_scalar_pred, scalar_coords = model(interface.coords)

    optimizer.zero_grad()
    loss = iloss.interface_l1_loss(interface_scalar_pred.squeeze(), interface.interface_data, method=prog_args.mse)
    loss.backward()

    grid_scalar_pred, grid_scalar_coords = model(sampled_grid_points)
    grid_scalar_grad = derivatives.gradient(grid_scalar_pred, grid_scalar_coords)
    grad_norm = torch.norm(grid_scalar_grad, p=2, dim=1)
    loss_grid = prog_args.lambda_g * torch.abs(grad_norm - 1).mean()
    loss_grid.backward()

    optimizer.step()
    scheduler.step()

    dur.append(time.time() - t0)
    print("Epoch {:05d} | Time(s) {:.4f} | Interface Loss {:.4f} | Norm Loss {:.4f}".format(epoch, dur[-1], loss.item(),
                                                                                            loss_grid.item()))

model_losses = {"L1 scalar constraints": loss.item()}

model.eval()
interface_scalar_pred, interface_coords = model(interface.coords)
interface = mm.get_horizon_metrics(interface_scalar_pred, interface_coords, interface)

grid_dict = {}
with torch.no_grad():
    for i in range(8):
        grid_piece = grid_dist.get_subset(i)
        grid_dist_coords = grid_piece.coords.to(0)
        scalar_pred, _ = model(grid_dist_coords)
        grid_piece.set_scalar_pred(scalar_pred.detach().cpu().numpy())
        grid_dict[i] = grid_piece

grid_points = combine_dist_grid_results(grid_dict, grid_points)

# Create Model Output
model_output = ModelOutput(prog_args, interface=interface, grid=grid_points,
                           model_metrics=model_losses)
model_output.output_model_and_metrics_to_file()

working_dir = prog_args.model_dir
pytorch_file = working_dir + '/model.pth'
torch.save(model.state_dict(), pytorch_file)