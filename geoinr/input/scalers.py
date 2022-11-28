import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class custom_scalar(object):
    def __init__(self, coords):
        self.coord_min = coords.min(axis=0)
        self.coord_max = coords.max(axis=0)

        self.coord_range = self.coord_max - self.coord_min
        range_max = self.coord_range.max()
        self.scale_range = np.array([range_max, range_max, self.coord_range[2]])
        self.scale_ = 1.0 / self.scale_range

    def transform(self, coords):
        # x_c = coords_center = coord_min + coord_range / 2
        # _____________
        # |            |
        # |            |
        # |            |
        # |    x_c     |   x_c = [x_c, y_c, z_c]
        # |            |
        # |            |
        # |            |
        # |            |
        # _____________
        # 1) subtract center from coords x_i = [x_i, y_i, z_i] => x_i - x_c = [x_i - x_c, y_i - y_c, z_i - z_c]
        # 2) divide coord from 1) by scaling factor scale_range
        # range_max : maximum range of all x, y, z ranges
        # scale_range : [range_max, range_max, coord_range[2] (z range)]
        #       this scales x/y isometrically, while z is different; ensures z coords are b/t 0 and 1. This is extremely
        #       useful for geological datasets where x/y ranges >>> z range. If z was also scaled by range_max
        #       (x/y/z isometric) and x/y ranges >>> z range model result would be chaos. This is b/c resulting scaled
        #       z range would be like [-0.0001, 0.0001] => z coords are essentially noise

        return (coords - (self.coord_min + self.coord_range / 2)) / (self.scale_range / 2)


class custom_scalar2(object):
    def __init__(self, coords):
        self.coord_min = coords.min(axis=0)
        self.coord_max = coords.max(axis=0)

        self.coord_range = self.coord_max - self.coord_min
        range_max = self.coord_range.max()
        self.scale_range = np.array([range_max, range_max, self.coord_range[2]])
        self.scale_ = 1.0 / self.scale_range

    def transform(self, coords):
        # x_c = coords_center = coord_min + coord_range / 2
        # _____________
        # |            |
        # |            |
        # |            |
        # |    x_c     |   x_c = [x_c, y_c, z_c]
        # |            |
        # |            |
        # |            |
        # |            |
        # _____________
        # 1) subtract center from coords x_i = [x_i, y_i, z_i] => x_i - x_c = [x_i - x_c, y_i - y_c, z_i - z_c]
        # 2) divide coord from 1) by scaling factor scale_range
        # range_max : maximum range of all x, y, z ranges
        # scale_range : [range_max, range_max, coord_range[2] (z range)]
        #       this scales x/y isometrically, while z is different; ensures z coords are b/t 0 and 1. This is extremely
        #       useful for geological datasets where x/y ranges >>> z range. If z was also scaled by range_max
        #       (x/y/z isometric) and x/y ranges >>> z range model result would be chaos. This is b/c resulting scaled
        #       z range would be like [-0.0001, 0.0001] => z coords are essentially noise

        return (coords - self.coord_min) / self.scale_range


class custom_scalar3(object):
    def __init__(self, coords, mininum=0.0, maximum=1.0):
        self.coord_min = coords.min(axis=0)
        self.coord_max = coords.max(axis=0)
        self.max = maximum
        self.min = mininum

        self.coord_range = self.coord_max - self.coord_min
        range_max = self.coord_range.max()
        self.scale_range = np.array([range_max, range_max, self.coord_range[2]])
        self.scale_ = 1.0 / self.scale_range

    def transform(self, coords):
        # x_c = coords_center = coord_min + coord_range / 2
        # _____________
        # |            |
        # |            |
        # |            |
        # |    x_c     |   x_c = [x_c, y_c, z_c]
        # |            |
        # |            |
        # |            |
        # |            |
        # _____________
        # 1) subtract center from coords x_i = [x_i, y_i, z_i] => x_i - x_c = [x_i - x_c, y_i - y_c, z_i - z_c]
        # 2) divide coord from 1) by scaling factor scale_range
        # range_max : maximum range of all x, y, z ranges
        # scale_range : [range_max, range_max, coord_range[2] (z range)]
        #       this scales x/y isometrically, while z is different; ensures z coords are b/t minimum and maximum. This is extremely
        #       useful for geological datasets where x/y ranges >>> z range. If z was also scaled by range_max
        #       (x/y/z isometric) and x/y ranges >>> z range model result would be chaos. This is b/c resulting scaled
        #       z range would be like [-0.0001, 0.0001] => z coords are essentially noise

        return ((coords - self.coord_min) / self.coord_range) * (self.max - self.min) + self.min


def get_data_scalar_from_coords(coords):
    scalar = StandardScaler()
    scalar.fit(coords)
    return scalar


def get_data_scalar_from_bounds(bounds):
    """ Given bounds (6D vector) [xmin, xmax, ymin, ymax, zmin, zmax] created the sklearn scalar object that will enable
    us to transform our coordinates to a normalized/scaled range. MinMaxScalar used. If StandardScalar is used we would
    have to do this differently; as it requires the entire dataset to get the mean and std."""

    domain_corner_points = np.zeros((8, 3))

    xmin = bounds[0]
    xmax = bounds[1]
    ymin = bounds[2]
    ymax = bounds[3]
    zmin = bounds[4]
    zmax = bounds[5]

    domain_corner_points[0] = [xmin, ymin, zmin]
    domain_corner_points[1] = [xmax, ymin, zmin]
    domain_corner_points[2] = [xmin, ymax, zmin]
    domain_corner_points[3] = [xmax, ymax, zmin]
    domain_corner_points[4] = [xmin, ymin, zmax]
    domain_corner_points[5] = [xmax, ymin, zmax]
    domain_corner_points[6] = [xmin, ymax, zmax]
    domain_corner_points[7] = [xmax, ymax, zmax]

    scalar = MinMaxScaler((-1, 1))
    scalar.fit(domain_corner_points)
    return scalar