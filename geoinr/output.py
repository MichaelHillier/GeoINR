import numpy as np
import pandas as pd
from geoinr.input.constraints.interface import InterfaceData, Series
from geoinr.input.constraints.unit import UnitData
from geoinr.input.grids import Grid
import pyvista as pv
from vtkmodules.all import vtkPolyData, vtkUnstructuredGrid, vtkStructuredGrid, vtkImageData, vtkPoints, vtkCellArray, \
    vtkXMLUnstructuredGridWriter, vtkXMLPolyDataWriter, vtkXMLImageDataWriter, vtkXMLStructuredGridWriter, \
    vtkCellDataToPointData, vtkContourFilter, vtkAppendPolyData
from geoinr.utils.vtk_utils import vertically_exaggerate_vtk_object
from geoinr.utils.plot_utils import build_plot_from_horizon_metrics, build_plot_from_unit_metrics


def write_polydata_file(filename: str, poly: vtkPolyData):
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(poly)
    writer.Update()


def write_unstructuredgrid_file(filename: str, mesh: vtkUnstructuredGrid):
    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(mesh)
    writer.Update()


def write_imagedata_file(filename: str, grid: vtkImageData):
    writer = vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Update()


def write_structuredgrid_file(filename: str, grid: vtkStructuredGrid):
    writer = vtkXMLStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Update()


def output_points_as_vtk_polydata(points, output_filename: str):
    points_vtk = vtkPoints()
    vertices = vtkCellArray()
    for pt in points:
        pt_id = points_vtk.InsertNextPoint(pt)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(pt_id)
    poly = vtkPolyData()
    poly.SetPoints(points_vtk)
    poly.SetVerts(vertices)

    write_polydata_file(output_filename, poly)


def extract_iso_surfaces_from_grid(grid, prop_name, iso_values, surface_indices=None):
    """
    :param grid: vtk grid to perform iso surface extraction
    :param prop_name: property name of vtkDataArray that will be used for contouring
    :param iso_values: iso-values/levels associated to modelled surfaces
    :param surface_indices: surface/horizon indices for modelled surfaces. This will be attributed
                            to the vtk geometry by creating a property called level. Each distinct
                            horizon will have its own level value (which is a horizon index). This
                            is very useful when dealing with scalar field series (multiple implicit
                            fields) and visualizing the model. Otherwise, the properties automatically
                            outputted from the contour filter can be poorly visualized because
                            1) different series can have overlapping ranges in values - surfaces
                               could be interpreted as being in multiple different geological histories
                               of deposition/erosion
                            2) resultant scalar field (combined from all series using knowledge)
                               property is corrupted on erosional surfaces (pts on either side have
                               different scalar ranges/values and their interpolation will create
                               property visualization issues in this zone.
                            e.g. iso_values = [4.5, 3.4] surface_indices = [6, 7].
                                 iso_surface (iso_value = 4.5) will have level = 6
                                 iso_surface (iso_value = 3.40 will have level = 7
    :return: iso surfaces (vtkPolyData)
    """
    # Set the array to be contoured
    index = grid.GetPointData().SetActiveScalars(prop_name)
    if index == -1:
        # Property maybe associated to CellData; try converting CellData to PointData
        c2p = vtkCellDataToPointData()
        c2p.SetInputData(grid)
        c2p.Update()
        grid = c2p.GetOutput()
        index = grid.GetPointData().SetActiveScalars(prop_name)
    assert index != -1, "Couldn't find data array with given property name"

    contour_filter = vtkContourFilter()
    contour_filter.SetInputData(grid)

    separated_interfaces = []
    if isinstance(iso_values, list) or isinstance(iso_values, np.ndarray):
        for i, iso_value in enumerate(iso_values):
            contour_filter.SetValue(i, iso_value)
            separate_contour_filter = vtkContourFilter()
            separate_contour_filter.SetInputData(grid)
            separate_contour_filter.SetValue(0, iso_value)
            separate_contour_filter.Update()
            seperated_interface = pv.PolyData(separate_contour_filter.GetOutput())
            separated_interfaces.append(seperated_interface)
        contour_filter.Update()
    else:
        contour_filter.SetValue(0, iso_values)
        contour_filter.Update()
        seperated_interface = pv.PolyData(contour_filter.GetOutput())
        separated_interfaces.append(seperated_interface)

    interfaces = contour_filter.GetOutput()

    if surface_indices is not None:
        # create new property on iso surfaces using horizon/surface indices.
        if isinstance(iso_values, np.ndarray):
            assert iso_values.size == len(surface_indices)
        elif isinstance(iso_values, list):
            assert len(iso_values) == len(surface_indices)
        else:
            assert len(surface_indices) == 1
        map_dict = {iso_values[i]: surface_indices[i] for i in range(len(surface_indices))}
        interfaces = pv.PolyData(interfaces)
        level = np.vectorize(map_dict.get)(interfaces.point_data[prop_name])
        interfaces.clear_point_data()
        interfaces.point_data['horizon_index'] = level

    # convert to pyvista PolyData
    interfaces = pv.PolyData(interfaces)

    return interfaces, separated_interfaces


def append_iso_surfaces(model_dir: str, surfaces, debug=False):
    if len(surfaces) == 1:
        return surfaces[0]
    else:
        appended_surfaces = vtkAppendPolyData()
        z = 0
        for surface in surfaces:
            assert isinstance(surface, vtkPolyData), "surface is not a vtkPolyData object"
            if debug:
                surf_filename = model_dir + '/surf' + str(z) + ".vtp"
                write_polydata_file(surf_filename, surface)
            appended_surfaces.AddInputData(surface)
            z += 1
        appended_surfaces.Update()
        return appended_surfaces.GetOutput()


def compute_distance_metrics_between_constraints_and_modelled_surfaces(
        interfaces: InterfaceData, surfaces: list, interfaces_test=None):

    #  Check interfaces and interfaces_test type
    if type(interfaces.interface_vtk) == vtkPolyData:
        # convert to pyvista data struct
        interfaces.interface_vtk = pv.PolyData(interfaces.interface_vtk)
    else:
        raise TypeError("Type of supplied interfaces object unexpected. Likely set to None.")
    if interfaces_test is not None:
        if type(interfaces_test.interface_vtk) == vtkPolyData:
            # convert to pyvista data struct
            interfaces_test.interface_vtk = pv.PolyData(interfaces_test.interface_vtk)
        else:
            raise TypeError("Type of supplied interfaces object unexpected. Likely set to None.")

    # Get unique constraint level values. Needs to match sequence with unique_horizon values.
    # May have to reverse the sequence to match. Depends on the level_data_mode (InterfaceData)
    # Three modes of extracting scalar constraints for interface data:
    #   1) level properties are in the correct order (largest numbers (youngest units) -> smallest numbers (oldest)),
    #            not normalized
    #   2) reverse level properties (smallest numbers (youngest units) -> largest numbers (oldest))
    #            not normalized
    #   3) level properties are in the correct order and IS normalized.
    # IF level_data_mode == 1 || 3 unique_constraint_levels have to be DESCENDING
    # IF level_data_mode == 2 unique_constraint_levels have to be ASCENDING
    if interfaces.level_data_mode == 2:
        unique_constraint_levels = np.unique(interfaces.interface_vtk.point_data['level'])
    else:
        unique_constraint_levels = np.unique(interfaces.interface_vtk.point_data['level'])[::-1]

    assert len(surfaces) == unique_constraint_levels.size, \
        "number of modelled horizons and unique level constraints do not match"

    horizon_distances_res = []
    horizon_distances_std = []
    test_horizon_distances_res = []
    test_horizon_distances_std = []

    for i, horizon_i_mesh in enumerate(surfaces):

        constraints_i = interfaces.interface_vtk.extract_points(
            interfaces.interface_vtk.point_data['level'] == unique_constraint_levels[i])
        constraints_i = pv.PolyData(constraints_i.points)
        _ = constraints_i.compute_implicit_distance(horizon_i_mesh, inplace=True)
        dist_i = np.abs(constraints_i['implicit_distance'])
        mean_dist_i = np.mean(dist_i)
        std_dist_i = np.std(dist_i)
        horizon_distances_res.append(mean_dist_i)
        horizon_distances_std.append(std_dist_i)

        if interfaces_test is not None:
            test_constraints_i = interfaces_test.interface_vtk.extract_points(
                interfaces_test.interface_vtk.point_data['level'] == unique_constraint_levels[i])
            test_constraints_i = pv.PolyData(test_constraints_i.points)
            _ = test_constraints_i.compute_implicit_distance(horizon_i_mesh, inplace=True)
            t_dist_i = np.abs(test_constraints_i['implicit_distance'])
            t_mean_dist_i = np.mean(t_dist_i)
            t_std_dist_i = np.std(t_dist_i)
            test_horizon_distances_res.append(t_mean_dist_i)
            test_horizon_distances_std.append(t_std_dist_i)

    horizon_distances_res = np.asarray(horizon_distances_res)
    horizon_distances_std = np.asarray(horizon_distances_std)
    interfaces.set_horizon_dist_residuals(horizon_distances_res)
    interfaces.set_horizon_dist_std(horizon_distances_std)

    if interfaces_test is not None:
        test_horizon_distances_res = np.asarray(test_horizon_distances_res)
        test_horizon_distances_std = np.asarray(test_horizon_distances_std)
        interfaces_test.set_horizon_dist_residuals(test_horizon_distances_res)
        interfaces_test.set_horizon_dist_std(test_horizon_distances_std)


def write_dict_data_to_file(dict_obj: dict, file_obj):
    for description, data in dict_obj.items():
        assert type(description) == str, "description (key) for dict is not a string"
        if isinstance(data, pd.DataFrame):
            # Compute summary statistics
            stats = data.describe(include='all')
            file_obj.write(description)
            file_obj.write("\n")
            file_obj.write(data.to_string(float_format='%.8f'))
            file_obj.write("\n")
            file_obj.write("Statistics:")
            file_obj.write("\n")
            file_obj.write(stats.to_string(float_format='%.8f'))
            file_obj.write("\n\n")
        elif isinstance(data, np.ndarray):
            file_obj.write(description)
            file_obj.write("\n")
            data_string = np.array2string(data, precision=4, separator=',', suppress_small=True)
            file_obj.write(data_string)
            file_obj.write("\n\n")
        elif isinstance(data, dict):
            file_obj.write(description)
            file_obj.write("\n")
            # create dataframe out of data (dict)
            df = pd.DataFrame({'keys': data.keys(), 'values': data.values()})
            file_obj.write(df.to_string(float_format='%.8f'))
            file_obj.write("\n\n")
        else:
            msg = "{:s} = {:.8f}\n".format(description, data)
            file_obj.write(msg)
    file_obj.write("\n\n")


class ModelOutput(object):
    def __init__(self, args, interface=None, orientation=None, unit=None, grid=None,
                 model_metrics=None, vertical_exaggeration=1, debug=False,
                 alternate_output_name=None, interface_test=None):
        self.args = args
        self.vertical_exaggeration = vertical_exaggeration
        self.debug = debug
        self.alternate_name = alternate_output_name

        self.interface_constraints = None
        self.iso_values = None
        self.interface_metrics = None
        self.interface_test_constraints = None
        self.interface_test_metrics = None

        self.normal_constraints = None
        self.normal_metrics = None

        self.unit_constraints = None
        self.unit_metrics = None
        self.map_to_original_class_ids = None

        self.model_metrics = model_metrics
        self.grid = None
        self.surfaces = []
        self.uncut_surfaces = []

        if interface is not None:
            assert interface.horizon_scalar_means is not None, "There are no horizon scalar means attributed to the" \
                                                               " interface data"
            self.iso_values = interface.horizon_scalar_means

        if orientation is not None:
            pass

        if unit is not None:
            self.__process_unit_input(unit)

        if grid is not None:
            self.__process_grid_input(grid)

        if self.uncut_surfaces:
            if interface is not None:
                compute_distance_metrics_between_constraints_and_modelled_surfaces(
                    interface,
                    self.uncut_surfaces,
                    interface_test)
                self.__process_interface_input(interface)
                if interface_test is not None:
                    self.__process_interface_test_input(interface_test)

    def __build_interface_metric_plot(self, interface: InterfaceData):
        scalar_means = None
        residual_means = None
        variance = None
        # get horizon metrics df
        horizon_metric_df = interface.interface_metric_dict['horizon_metrics']
        if 'scalar_means' in horizon_metric_df:
            scalar_means = horizon_metric_df['scalar_means']
        if 'residual_means' in horizon_metric_df:
            residual_means = horizon_metric_df['residual_means']
        if 'variance' in horizon_metric_df:
            variance = horizon_metric_df['variance']
        filename = self.args.model_dir + "/interface_metrics.png"
        build_plot_from_horizon_metrics(scalar_means, residual_means, variance, filename)

    def __process_interface_input(self, interface: InterfaceData):
        interface.process_model_outputs()
        if interface.interface_vtk is not None:
            self.add_interface_constraints(interface.interface_vtk)
        if interface.interface_metric_dict is not None:
            self.interface_metrics = interface.interface_metric_dict
            self.__build_interface_metric_plot(interface)

    def __process_interface_test_input(self, interface_test: InterfaceData):
        interface_test.process_model_outputs()
        if interface_test.interface_vtk is not None:
            self.add_interface_test_constraints(interface_test.interface_vtk)
        if interface_test.interface_metric_dict is not None:
            self.interface_test_metrics = interface_test.interface_metric_dict

    def __build_unit_metric_plot(self, unit: UnitData):
        class_ids = None
        class_residual_means = None
        # get class residual means df
        class_residual_mean_df = unit.unit_metric_dict['class_residual_means']
        if 'class_id' in class_residual_mean_df:
            class_ids = class_residual_mean_df['class_id']
        if 'residual_mean' in class_residual_mean_df:
            class_residual_means = class_residual_mean_df['residual_mean']
        filename = self.args.model_dir + "/unit_metrics.png"
        build_plot_from_unit_metrics(class_ids, class_residual_means, filename)

    def __process_unit_input(self, unit: UnitData):
        unit.process_model_outputs()
        self.map_to_original_class_ids = unit.unit_id_to_unit_level
        if unit.unit_vtk is not None:
            self.add_unit_constraints(unit.unit_vtk)
        if unit.unit_metric_dict is not None:
            self.add_unit_metrics(unit.unit_metric_dict)
            self.__build_unit_metric_plot(unit)

    def __process_grid_input(self, grid: Grid):
        grid.process_model_outputs(self.map_to_original_class_ids)
        self.add_model_grid(grid.grid_vtk)
        if self.iso_values is not None:
            # see Grid: __add_properties_to_vtk_object_if_present() for property naming scheme
            if grid.scalar_series is None:
                self.generate_iso_surfaces("Scalar Field", self.iso_values)
            else:
                n_series = grid.scalar_series.shape[1]
                for i in range(n_series):
                    series_iso_values = self.iso_values[grid.series.series_dict[i]]
                    scalar_property_name = "Scalar Field" + str(i)
                    self.generate_iso_surfaces(scalar_property_name, series_iso_values, grid.series.series_dict[i])
                # write out un-cutted surfaces to disk if debug = True
                for i, surface in enumerate(self.uncut_surfaces):
                    if self.debug:
                        if self.alternate_name is None:
                            surf_filename = self.args.model_dir + '/uncut_surf' + str(i) + ".vtp"
                        else:
                            surf_filename = self.args.model_dir + '/' + self.alternate_name + '_uncut_surf' + str(i) + ".vtp"
                        write_polydata_file(surf_filename, surface)
                self.cut_series_iso_surfacesV2(grid.series)
            self.surfaces = append_iso_surfaces(self.args.model_dir, self.surfaces, self.debug)

    def add_model_grid(self, grid):
        if self.vertical_exaggeration != 1:
            assert self.vertical_exaggeration > 1, "vertical exaggeration variable not set to a number greater than 1"
            grid = vertically_exaggerate_vtk_object(grid, self.vertical_exaggeration)
        self.grid = grid

    def add_model_metrics(self, model_metrics):
        self.model_metrics = model_metrics

    def add_interface_test_constraints(self, interface_test_constraints):
        if self.vertical_exaggeration != 1:
            assert self.vertical_exaggeration > 1, "vertical exaggeration variable not set to a number greater than 1"
            interface_test_constraints = vertically_exaggerate_vtk_object(interface_test_constraints, self.vertical_exaggeration)
        self.interface_test_constraints = interface_test_constraints

    def add_interface_constraints(self, interface_constraints):
        if self.vertical_exaggeration != 1:
            assert self.vertical_exaggeration > 1, "vertical exaggeration variable not set to a number greater than 1"
            interface_constraints = vertically_exaggerate_vtk_object(interface_constraints, self.vertical_exaggeration)
        self.interface_constraints = interface_constraints

    def add_normal_constraints(self, normal_constraints):
        if self.vertical_exaggeration != 1:
            assert self.vertical_exaggeration > 1, "vertical exaggeration variable not set to a number greater than 1"
            normal_constraints = vertically_exaggerate_vtk_object(normal_constraints, self.vertical_exaggeration)
        self.normal_constraints = normal_constraints

    def add_normal_metrics(self, normal_metrics):
        self.normal_metrics = normal_metrics

    def add_unit_constraints(self, unit_constraints):
        if self.vertical_exaggeration != 1:
            assert self.vertical_exaggeration > 1, "vertical exaggeration variable not set to a number greater than 1"
            unit_constraints = vertically_exaggerate_vtk_object(unit_constraints, self.vertical_exaggeration)
        self.unit_constraints = unit_constraints

    def add_unit_metrics(self, unit_metrics):
        self.unit_metrics = unit_metrics

    def generate_iso_surfaces(self, scalar_array_name, iso_values, surface_indices=None):
        assert self.grid is not None, "There is not grid to extract iso surfaces from"
        # Below for when ModelOutput is built manually (by individually calling methods in driver code).
        # If automation is used via ModelOutput initialization (giving all object up front) iso_values is already set.
        # We don't want to override it, since when dealing with scalar series will cause big issues.
        if self.iso_values is None:
            self.iso_values = iso_values
        interfaces, separated_interfaces = extract_iso_surfaces_from_grid(self.grid, scalar_array_name, iso_values, surface_indices)
        self.surfaces.append(interfaces)
        for separated_interface in separated_interfaces:
            self.uncut_surfaces.append(separated_interface)
        # don't need to vertical exaggeration (if vertical_exaggeration != 1) since grid was already exaggerated;
        # resulting iso surface would be exaggerated by default

    def cut_series_iso_surfaces(self, series: Series):
        """
        :param series: contains all the series data/information
        This function keeps basement continuous (younger unconformities can't cut basement). This was behavior requested
        by Karine. I don't think this is the best approach. V2 below is better imo.
        Cuts isosurfaces, each described by their own scalar field, in a manner respecting the geological history.
        Use pyvista's clip_surface()
        If a surface a is being clipped by surface b below (older) a.clip_surface(b)
        If a surface a is being clipped by surface b above (younger) a.clip_surface(b, invert=False)
        There is a lot of complication related to keeping the basement continuous which is why there is so much code.
        """

        n_series = len(self.surfaces)
        for i in range(n_series):
            if type(self.surfaces[i]) != pv.PolyData:
                raise TypeError("Surface a not a pyvista PolyData when cutting iso_surfaces")

        unconformity_series_ids = series.get_unconformity_series_ids()
        # cut all unconformities except basement by basement unconformity
        for i in range(unconformity_series_ids.size - 1):
            clipped_surface = self.surfaces[unconformity_series_ids[i]].clip_surface(
                self.surfaces[unconformity_series_ids[-1]])
            if clipped_surface.n_points != 0 and clipped_surface.n_cells != 0:
                self.surfaces[unconformity_series_ids[i]] = clipped_surface

        # cut all unconformities except basement by younger unconformities
        unconformity_series_ids = unconformity_series_ids[::-1]
        unconformity_series_ids_to_process = unconformity_series_ids[1:]
        for u_series_id in unconformity_series_ids_to_process:
            # get series_ids of unconformities younger than u_series_id arranged older to younger
            # so that we progressively cut current unconformity by younger unconformities (how these are cut
            # naturally in the geological series of events)
            younger_u_series_ids = series.get_unconformity_series_ids_younger_than(u_series_id)
            for younger_u_series_id in younger_u_series_ids:
                clipped_surface = self.surfaces[u_series_id].clip_surface(self.surfaces[younger_u_series_id],
                                                                          invert=False)
                if clipped_surface.n_points != 0 and clipped_surface.n_cells != 0:
                    self.surfaces[u_series_id] = clipped_surface

        # cut all onlap scalar fields (onlap_series_id) by unconformities younger than onlap_series_id
        onlap_series_ids = series.get_onlap_series_ids()[::-1]
        for onlap_series_id in onlap_series_ids:
            # get series_ids of unconformities younger than onlap_series_id arranged older to younger
            # so that we progressively cut onlap series by unconformities younger than onlap_series_id
            younger_u_series_ids = series.get_unconformity_series_ids_younger_than(onlap_series_id)
            for younger_u_series_id in younger_u_series_ids:
                clipped_surface = self.surfaces[onlap_series_id].clip_surface(self.surfaces[younger_u_series_id],
                                                                              invert=False)
                if clipped_surface.n_points != 0 and clipped_surface.n_cells != 0:
                    self.surfaces[onlap_series_id] = clipped_surface
            # cut surface from unconformities below
            older_u_series_ids = series.get_unconformity_series_ids_below(onlap_series_id)
            # cut only the unconformity below onlap_series_id AND the last unconformity (basement)
            older_u_series_ids = np.array([older_u_series_ids[0], older_u_series_ids[-1]])
            for older_u_series_id in older_u_series_ids:
                clipped_surface = self.surfaces[onlap_series_id].clip_surface(self.surfaces[older_u_series_id])
                if clipped_surface.n_points != 0 and clipped_surface.n_cells != 0:
                    self.surfaces[onlap_series_id] = clipped_surface

    def cut_series_iso_surfacesV2(self, series: Series):
        """
        :param series: contains all the series data/information
        In this function, any unconformity can cut anything that is older. Best approach IMO.
        Cuts isosurfaces, each described by their own scalar field, in a manner respecting the geological history.
        Use pyvista's clip_surface()
        If a surface a is being clipped by surface b below (older) a.clip_surface(b)
        If a surface a is being clipped by surface b above (younger) a.clip_surface(b, invert=False)
        There is a lot of complication related to keeping the basement continuous which is why there is so much code.
        """

        n_series = len(self.surfaces)
        for i in range(n_series):
            if type(self.surfaces[i]) != pv.PolyData:
                raise TypeError("Surface a not a pyvista PolyData when cutting iso_surfaces")

        unconformity_series_ids = series.get_unconformity_series_ids()

        # get unconformity series ids arranged older to younger
        unconformity_series_ids = unconformity_series_ids[::-1]
        for u_series_id in unconformity_series_ids:
            # get series_ids of unconformities younger than u_series_id arranged older to younger
            # so that we progressively cut current unconformity by younger unconformities (how these are cut
            # naturally in the geological series of events)
            younger_u_series_ids = series.get_unconformity_series_ids_younger_than(u_series_id)
            for younger_u_series_id in younger_u_series_ids:
                clipped_surface = self.surfaces[u_series_id].clip_surface(self.surfaces[younger_u_series_id],
                                                                          invert=False)
                if clipped_surface.n_points != 0 and clipped_surface.n_cells != 0:
                    self.surfaces[u_series_id] = clipped_surface

        # cut all onlap scalar fields (onlap_series_id) by unconformities younger than onlap_series_id
        onlap_series_ids = series.get_onlap_series_ids()[::-1]
        for onlap_series_id in onlap_series_ids:
            # get series_ids of unconformities younger than onlap_series_id arranged older to younger
            # so that we progressively cut onlap series by unconformities younger than onlap_series_id
            younger_u_series_ids = series.get_unconformity_series_ids_younger_than(onlap_series_id)
            for younger_u_series_id in younger_u_series_ids:
                clipped_surface = self.surfaces[onlap_series_id].clip_surface(self.surfaces[younger_u_series_id],
                                                                              invert=False)
                if clipped_surface.n_points != 0 and clipped_surface.n_cells != 0:
                    self.surfaces[onlap_series_id] = clipped_surface
            # cut surface from unconformities below
            older_u_series_ids = series.get_unconformity_series_ids_below(onlap_series_id)
            # cut only the unconformity below onlap_series_id AND the last unconformity (basement)
            older_u_series_ids = np.array([older_u_series_ids[0], older_u_series_ids[-1]])
            for older_u_series_id in older_u_series_ids:
                clipped_surface = self.surfaces[onlap_series_id].clip_surface(self.surfaces[older_u_series_id])
                if clipped_surface.n_points != 0 and clipped_surface.n_cells != 0:
                    self.surfaces[onlap_series_id] = clipped_surface

    def write_model_args_and_metrics_to_file(self):
        filename = self.args.model_dir + "/model.txt"
        if self.alternate_name is not None:
            filename = self.args.model_dir + "/model_" + self.alternate_name + ".txt"
        # create file
        f = open(filename, "w")
        # 1) Write model args/paramaeters
        params = vars(self.args)
        f.write("Model Parameters:\n")
        for parameter, value in params.items():
            msg = '{param} = {value}\n'.format(param=parameter, value=value)
            f.write(msg)
        f.write("\n\n")
        # 2) Write model metrics
        if self.model_metrics is not None:
            f.write("Model Metrics:\n")
            write_dict_data_to_file(self.model_metrics, f)
        # 3) Write mean iso values for each horizon
        if self.iso_values is not None:
            f.write("Mean Horizon Isovalues:\n")
            for i, value in enumerate(self.iso_values):
                msg = "{:d} = {:.3f}\n".format(i, value)
                f.write(msg)
            f.write("\n\n")
        # 4) Write interface metrics; e.g. mean iso value, dist residual, grad norm, loss
        if self.interface_metrics is not None:
            f.write("Interface Train Metrics: \n")
            write_dict_data_to_file(self.interface_metrics, f)
        if self.interface_test_metrics is not None:
            f.write("Interface Test Metrics: \n")
            write_dict_data_to_file(self.interface_test_metrics, f)
        # 5) Write normal observations metrics
        if self.normal_metrics is not None:
            f.write("Normal Metrics: \n")
            write_dict_data_to_file(self.normal_metrics, f)
        # 6) Write unit observation metrics
        if self.unit_metrics is not None:
            f.write("Unit Metrics: \n")
            write_dict_data_to_file(self.unit_metrics, f)
        f.close()

    def output_model_and_metrics_to_file(self, output_grid=True):
        if self.alternate_name is not None:
            base_filename = self.args.model_dir + '/' + self.alternate_name
        else:
            base_filename = self.generate_filename_from_parameters()
        if output_grid:
            if type(self.grid) == vtkUnstructuredGrid:
                write_unstructuredgrid_file(base_filename + '_grid.vtu', self.grid)
            elif type(self.grid) == vtkStructuredGrid:
                write_structuredgrid_file(base_filename + '_grid.vts', self.grid)
            elif type(self.grid) == vtkImageData:
                write_imagedata_file(base_filename + '_grid.vti', self.grid)
            else:
                raise ValueError("grid is not a supported vtk volume")
        if self.surfaces:
            if isinstance(self.surfaces, pv.PolyData):
                self.surfaces.save(base_filename + '_surf.vtp')
            else:
                write_polydata_file(base_filename + '_surf.vtp', self.surfaces)
        if self.interface_constraints is not None:
            if isinstance(self.interface_constraints, pv.PolyData):
                self.interface_constraints.save(base_filename + '_interface.vtp')
            else:
                write_polydata_file(base_filename + '_interface.vtp', self.interface_constraints)
        if self.interface_test_constraints is not None:
            if isinstance(self.interface_test_constraints, pv.PolyData):
                self.interface_test_constraints.save(base_filename + '_interface_test.vtp')
            else:
                write_polydata_file(base_filename + '_interface_test.vtp', self.interface_constraints)
        if self.normal_constraints is not None:
            if isinstance(self.normal_constraints, pv.PolyData):
                self.normal_constraints.save(base_filename + '_normal.vtp')
            else:
                write_polydata_file(base_filename + '_normal.vtp', self.normal_constraints)
        if self.unit_constraints is not None:
            if isinstance(self.unit_constraints, pv.PolyData):
                self.unit_constraints.save(base_filename + '_units.vtp')
            else:
                write_polydata_file(base_filename + '_units.vtp', self.unit_constraints)
        self.write_model_args_and_metrics_to_file()

    def generate_filename_from_parameters(self):
        dataset = self.args.dataset.replace('/', '_')
        filename = self.args.model_dir + '/' + dataset
        if self.args.technique == "sparse_adj_matrix":
            filename += '_' + 'sadj'
            filename += '_' + self.args.convolution
        elif self.args.technique == 'pytorch_geometric':
            filename += '_' + 'pg'
            filename += '_' + self.args.convolution
        elif self.args.technique == 'mlp':
            filename += '_' + 'mlp'
        else:
            ValueError("Unknown technique type")
        filename += '_' + self.args.activation
        filename += '_' + self.args.mse
        if self.args.global_anisotropy:
            filename += '_gp'
        if self.args.mtl:
            filename += '_mtl'
        filename += '_emb' + str(self.args.embed_dim)
        filename += '_nl' + str(self.args.num_hidden_layers)
        filename += '_ep' + str(self.args.num_epocs)
        return filename
