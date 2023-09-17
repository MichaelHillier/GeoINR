import os
import socket
from datetime import datetime
import torch
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import geoinr.args as args
from geoinr.input.scalers import get_scaler
from geoinr.input.constraints.interface import InterfaceData, InterfaceKFoldSplit
from geoinr.input.grids import Grid, GridPointDataDistributedSampler, GridData, \
    generate_vtk_structure_grid_and_grid_points
from geoinr.utils.vtk_utils import get_resultant_bounds_from_vtk_objects
from geoinr.utils.distributed import combine_dist_grid_results
from geoinr.utils.model_processing import cut_series_by_boundary_surfaces
from geoinr.model.mlp.model import SeriesMLP
from geoinr.output import ModelOutput
import geoinr.loss.orientation as oloss
import geoinr.loss.horizon as hloss
import geoinr.model_metrics as mm


class StepLearningRateSchedule(object):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.get_learning_rate(epoch)


prog_args = args.arg_parse()
options = vars(prog_args)

lr_scheduler = StepLearningRateSchedule(prog_args.learning_rate, 500, 0.5)

# Load Data
# Get Interface Data
interface_file = os.path.join(args.ROOT_DIR, prog_args.datadir, prog_args.dataset, 'marker_subset.vtp')
interface_info_file = os.path.join(args.ROOT_DIR, prog_args.datadir, prog_args.dataset, 'marker_info.csv')
interface = InterfaceData(interface_file, 2, interface_info_file, prog_args.efficient,
                          prog_args.youngest_unit_sampled)

# get data bounds from combined datasets; interface, orientation, classes. Used for feature normalization/scaling
bounds = get_resultant_bounds_from_vtk_objects(interface.interface_vtk,
                                               xy_buffer=prog_args.xy_buffer,
                                               z_buffer=prog_args.z_buffer)
# get normalization/scaling function
scalar = get_scaler(prog_args.scale_method, interface.coords, prog_args.scale_range)

if prog_args.kfold > 0:
    kfold = InterfaceKFoldSplit(interface, 2, True)

# normalize/scale features
interface.transform(scalar)
interface.convert_to_torch()
interface.send_to_gpu(0)

series_dict = interface.series.series_dict
unconformity_dict = interface.series.unconformity_dict
n_series = len(series_dict)

# Get Grid Point Data
# Create grid points first
grid_coords, grid = generate_vtk_structure_grid_and_grid_points(bounds, prog_args.xy_resolution, prog_args.z_resolution)
grid_points = Grid(grid_coords, grid, interface.series)
# normalize/scale grid features
grid_points.transform(scalar)
# split/chunk evaluation/inference data for distributed computing
grid_dist = GridPointDataDistributedSampler(grid_points, 32)
grid_points.uniformly_sample_grid_for_points(prog_args.n_grid_samples)
grid_points.transform_sampled_grid_points(scalar)
sampled_grid_points = torch.from_numpy(grid_points.sampled_grid_points).float().contiguous().to(0)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join(args.ROOT_DIR, 'runs', current_time + '_' + socket.gethostname())
writer = SummaryWriter(log_dir=log_dir)
prog_args.model_dir = writer.log_dir

# Create INR model
weights_file = args.get_pretrained_model_file('plane', prog_args)
model = SeriesMLP(3, prog_args, n_series, weights_file).to(0)
optimizer = torch.optim.AdamW(model.parameters(), lr=prog_args.learning_rate, weight_decay=prog_args.weight_decay,
                              amsgrad=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=prog_args.num_epocs, eta_min=0,
                                                       last_epoch=-1, verbose=False)

dur = []
# loss_values = []
for epoch in range(prog_args.num_epocs):
    model.train()
    t0 = time.time()
    # forward
    interface_scalar_pred, scalar_coords = model(interface.coords)
    loss_h_residuals, loss_h_var, loss_h_pol, h_mean = \
        hloss.horizon_loss_multiple_series(interface_scalar_pred,
                                           scalar_coords,
                                           interface.horizon_interface_indices,
                                           series_dict)
    interface.series.set_mean_scalar_values_for_series(h_mean)

    loss_strati_above_below = hloss.stratigraphic_above_below_losses(interface_scalar_pred, scalar_coords,
                                                                     interface.horizon_interface_indices,
                                                                     interface.series)

    optimizer.zero_grad()
    loss_constraints = loss_h_residuals.sum() + loss_h_var.sum()

    loss_consistency = loss_strati_above_below.sum()

    loss_constraints_and_consistency = loss_constraints + loss_consistency
    loss_constraints_and_consistency.backward()

    grid_scalar_pred, grid_scalar_coords = model(sampled_grid_points)
    loss_norm = prog_args.lambda_g * oloss.norm_loss_multiple_series(grid_scalar_pred, grid_scalar_coords, series_dict)
    loss_global = loss_norm.sum()
    loss_global.backward()

    optimizer.step()
    scheduler.step()

    dur.append(time.time() - t0)
    print("Epoch {:05d} | Time(s) {:.4f} | Horizon Loss {:.4f} | Variance Loss {:.4f}"
          " | Polarity Loss {:.4f} | Norm Loss {:.4f} | Consistency {:.4f}".format(
        epoch, dur[-1], loss_h_residuals.sum().item(),
        loss_h_var.sum().item(), loss_h_pol.sum().item(), loss_norm.sum().item(),
        loss_consistency.item()))


model_losses = {"interface_residuals": loss_h_residuals.sum().item(),
                "horizon_variance": loss_h_var.sum().item(),
                "horizon_polarity": loss_h_pol.sum().item(),
                "norm": loss_norm.sum().item(),
                "consistency": loss_consistency.sum().item()}

model.eval()
interface_scalar_pred, interface_coords = model(interface.coords)
interface = mm.get_horizon_metrics_for_multiple_series(interface_scalar_pred, interface_coords, interface, series_dict)
horizon_scalar_means = interface.horizon_scalar_means
horizon_scalar_means = torch.from_numpy(horizon_scalar_means).float().to(0)
interface.series.set_mean_scalar_values_for_series(horizon_scalar_means)
unconformity_dict = {series_id: horizon_scalar_means[horizon_id] for series_id, horizon_id in unconformity_dict.items()}


grid_dict = {}
with torch.no_grad():
    for i in range(32):
        grid_piece = grid_dist.get_subset(i)
        grid_dist_coords = grid_piece.coords.to(0)
        scalar_pred, _ = model(grid_dist_coords)
        scalar_resultant, scalar_pred, unit_pred = cut_series_by_boundary_surfaces(scalar_pred, interface.series)
        grid_piece.set_scalar_pred(scalar_resultant.detach().cpu().numpy())
        grid_piece.set_scalar_series(scalar_pred.detach().cpu().numpy())
        grid_piece.set_unit_pred(unit_pred.detach().cpu().numpy())
        grid_dict[i] = grid_piece

grid_points = combine_dist_grid_results(grid_dict, grid_points)

# Create Model Output
model_output = ModelOutput(prog_args, interface=interface, grid=grid_points,
                           model_metrics=model_losses)
model_output.output_model_and_metrics_to_file()

if prog_args.kfold > 0:
    # KFold Cross Validation Procedure
    k = 0
    interface_residual_mean_k = []
    for (interface_k_train, interface_k_test) in kfold.split:
        # normalize/scale features
        interface_k_train.transform(scalar)
        interface_k_train.convert_to_torch()
        interface_k_train.send_to_gpu(0)

        # Create INR model
        model_k = SeriesMLP(3, prog_args.embed_dim, prog_args.num_hidden_layers, n_series, prog_args.concat, weights_file).to(0)
        optimizer_k = torch.optim.AdamW(model_k.parameters(), lr=prog_args.learning_rate, weight_decay=prog_args.weight_decay,
                                        amsgrad=True)

        scheduler_k = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_k, T_max=prog_args.num_epocs, eta_min=0,
                                                                 last_epoch=-1, verbose=False)

        dur = []
        for epoch in range(prog_args.num_epocs):
            model_k.train()
            t0 = time.time()
            # forward
            interface_scalar_pred, scalar_coords = model_k(interface_k_train.coords)
            loss_h_residuals, loss_h_var, loss_h_pol, h_mean = \
                hloss.horizon_loss_multiple_series(interface_scalar_pred,
                                                   scalar_coords,
                                                   interface_k_train.horizon_interface_indices,
                                                   series_dict)
            interface_k_train.series.set_mean_scalar_values_for_series(h_mean)

            loss_strati_above_below = hloss.stratigraphic_above_below_losses(interface_scalar_pred, scalar_coords,
                                                                             interface_k_train.horizon_interface_indices,
                                                                             interface_k_train.series)

            optimizer_k.zero_grad()
            loss_constraints = loss_h_residuals.sum() + loss_h_var.sum()

            loss_consistency = loss_strati_above_below.sum()

            loss_constraints_and_consistency = loss_constraints + loss_consistency
            loss_constraints_and_consistency.backward()

            grid_scalar_pred, grid_scalar_coords = model(sampled_grid_points)
            loss_norm = 0.1 * oloss.norm_loss_multiple_series(grid_scalar_pred, grid_scalar_coords, series_dict)
            loss_global = loss_norm.sum()
            loss_global.backward()

            optimizer_k.step()
            scheduler_k.step()

            dur.append(time.time() - t0)
            print("Epoch {:05d} | Time(s) {:.4f} | Horizon Loss {:.4f} | Variance Loss {:.4f}"
                  " | Polarity Loss {:.4f} | Norm Loss {:.4f} | Consistency {:.4f}".format(
                epoch, dur[-1], loss_h_residuals.sum().item(),
                loss_h_var.sum().item(), loss_h_pol.sum().item(), loss_norm.sum().item(),
                loss_consistency.item()))

        model_losses = {"interface_residuals": loss_h_residuals.sum().item(),
                        "horizon_variance": loss_h_var.sum().item(),
                        "horizon_polarity": loss_h_pol.sum().item(),
                        "norm": loss_norm.sum().item(),
                        "consistency": loss_consistency.sum().item()}

        # Evaluation on test set
        model_k.eval()
        # normalize/scale features for test set
        interface_k_test.transform(scalar)
        interface_k_test.convert_to_torch()
        interface_k_test.send_to_gpu(0)
        # adding metrics to k-th test data for cross validation
        interface_test_scalar_pred, test_interface_coords = model_k(interface_k_test.coords)
        interface_k_test = mm.get_horizon_metrics_for_multiple_series(interface_test_scalar_pred, test_interface_coords,
                                                                      interface_k_test, series_dict)

        # adding metrics to k-th training data for cross validation
        interface_train_scalar_pred, train_interface_coords = model_k(interface_k_train.coords)
        interface_k_train = mm.get_horizon_metrics_for_multiple_series(interface_train_scalar_pred, train_interface_coords,
                                                                       interface_k_train, series_dict)
        horizon_scalar_means_k = interface_k_train.horizon_scalar_means
        horizon_scalar_means_k = torch.from_numpy(horizon_scalar_means_k).float().to(0)
        interface_k_train.series.set_mean_scalar_values_for_series(horizon_scalar_means_k)

        grid_dict = {}
        with torch.no_grad():
            for i in range(32):
                grid_piece = grid_dist.get_subset(i)
                grid_dist_coords = grid_piece.coords.to(0)
                scalar_pred, _ = model_k(grid_dist_coords)
                scalar_resultant, scalar_pred, unit_pred = cut_series_by_boundary_surfaces(scalar_pred, interface_k_train.series)
                grid_piece.set_scalar_pred(scalar_resultant.detach().cpu().numpy())
                grid_piece.set_scalar_series(scalar_pred.detach().cpu().numpy())
                grid_piece.set_unit_pred(unit_pred.detach().cpu().numpy())
                grid_dict[i] = grid_piece

        grid_points = combine_dist_grid_results(grid_dict, grid_points)

        # Create Model Output
        model_output_k = ModelOutput(prog_args, interface=interface_k_train, grid=grid_points,
                                     model_metrics=model_losses,
                                     alternate_output_name=str(k), interface_test=interface_k_test, debug=True)
        model_output_k.output_model_and_metrics_to_file()

        # Compute the mean distance residuals over all modelled horizons
        interface_residual_mean_k.append(interface_k_test.horizon_dist_residuals.mean())
        # Compute the mean distance residuals STD over all modelled horizons

        k += 1

    mean_distance_residuals_of_all_test_splits = np.asarray(interface_residual_mean_k).mean()
    print("Mean distance residuals from all splits= ", mean_distance_residuals_of_all_test_splits)
