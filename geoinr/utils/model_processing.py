import torch
from geoinr.input.constraints import series


def cut_series_by_boundary_surfaces(scalar_fields: torch.Tensor, series_struct: series.Series):
    """
    This function cuts the series of scalar fields, describing distinct geological features,
    using the appropriate logic derived from the geological knowledge defined from the inputted
    relationship information for each geological interface.
    scalar_fields: [N, n_series]
    series: contains all the geological knowledge about the geological features
    """
    n_grid = scalar_fields.size()[0]
    # tracks the points that are processed (assigned to a valid geological domain)
    processed = torch.zeros(n_grid, dtype=torch.bool).to(scalar_fields.device)
    # Fill the following vectors with the model data, in a way the respects the geological knowledge
    resultant_scalar = torch.ones(n_grid).to(scalar_fields.device)
    units = torch.ones(n_grid).long()
    # This object contains all the meta data for the geological knowledge need to find/carve out
    # all the valid geological domains associated to each scalar field
    domains = series_struct.geological_domains
    # processing domains in older to younger sequence. domain data is structured that way
    for i in range(domains.n_domains):
        above_binterface = domains.above_boundary_interface[i]
        above_binterface_s_id = domains.above_boundary_interface_s_id[i]
        if above_binterface == -1:  # For last domain, domain_region is above youngest boundary/unconformity
            below_binterface = domains.above_boundary_interface[i - 1]
            below_binterface_s_id = domains.above_boundary_interface_s_id[i - 1]
            above_b_itrf = \
                scalar_fields[:, below_binterface_s_id] > series_struct.mean_scalar_values_for_series[below_binterface]
            domain_region = above_b_itrf & ~processed
        else:
            below_b_itrf = scalar_fields[:, above_binterface_s_id] < series_struct.mean_scalar_values_for_series[above_binterface]
            domain_region = below_b_itrf & ~processed
        younger_cutting_interfaces = domains.cutting_boundaries[i]
        if younger_cutting_interfaces:
            # carve domain region if there are boundaries/unconformites that cut/erode domain_region
            for j, cutting_interface in enumerate(younger_cutting_interfaces):
                # cutting_interface must be arranged older to younger
                cutting_interface_s_id = domains.cutting_boundaries_s_id[i][j]
                below_y_cut_itrf = \
                    scalar_fields[:, cutting_interface_s_id] < series_struct.mean_scalar_values_for_series[cutting_interface]
                below_y_cut_itrf_and_prev_true = below_y_cut_itrf & domain_region
                domain_region = below_y_cut_itrf_and_prev_true
        # Set scalar field to valid geological domain
        domain_s_id = domains.series_id[i]
        current_scalar_field = scalar_fields[:, domain_s_id]
        resultant_scalar[domain_region] = current_scalar_field[domain_region]
        # If the domain's scalar field is a conformal series we must iterate
        # over each interface within it the define all the units
        conformal_interfaces = domains.conformal_interfaces[i]
        if conformal_interfaces:
            # conformal_interfaces must be arranged older to younger
            n_conformal_interfaces = len(conformal_interfaces)
            for j, conformal_interface in enumerate(conformal_interfaces):
                below_c_itrf = scalar_fields[:, domain_s_id] < series_struct.mean_scalar_values_for_series[conformal_interface]
                below_true = torch.any(below_c_itrf)
                below_c_itrf_and_in_domain = below_c_itrf & domain_region
                below_in_domain = torch.any(below_c_itrf_and_in_domain)
                below_c_itrf_and_in_domain_unprocessed = below_c_itrf_and_in_domain & ~processed
                below_in_domain_unp = torch.any(below_c_itrf_and_in_domain_unprocessed)
                units[below_c_itrf_and_in_domain_unprocessed] = domains.units[i][j]
                processed[below_c_itrf_and_in_domain_unprocessed] = True
            # process above youngest interface
            above_y_itrf = domain_region & ~processed
            units[above_y_itrf] = conformal_interfaces[n_conformal_interfaces - 1]
            processed[above_y_itrf] = True
        else:
            units[domain_region] = domains.units[i][0]
            processed[domain_region] = True

    return resultant_scalar, scalar_fields, units






