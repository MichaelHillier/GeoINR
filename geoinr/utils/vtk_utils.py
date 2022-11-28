import numpy as np
import math
from vtkmodules.all import vtkImageData, vtkStructuredGrid, vtkUnstructuredGrid, vtkPolyData, vtkTransform, \
    vtkTransformFilter, vtkBoundingBox, vtkDataSet, VTK_DOUBLE, VTK_INT, vtkLookupTable, vtkColorTransferFunction, \
    vtkImagePermute, vtkProbeFilter, vtkImageMapToColors, vtkPNGWriter, vtkCellArray, vtkPoints
from vtkmodules.util import numpy_support


def create_continuous_property_vtk_array(name: str, arr: np.ndarray):
    vtk_arr = numpy_support.numpy_to_vtk(arr, deep=True, array_type=VTK_DOUBLE)
    vtk_arr.SetName(name)
    if arr.ndim == 2 and arr.shape[1] != 1:
        vtk_arr.SetNumberOfComponents(arr.shape[1])
    return vtk_arr


def create_discrete_property_vtk_array(name: str, arr: np.ndarray):
    vtk_arr = numpy_support.numpy_to_vtk(arr, deep=True, array_type=VTK_INT)
    vtk_arr.SetName(name)
    if arr.ndim == 2 and arr.shape[1] != 1:
        vtk_arr.SetNumberOfComponents(arr.shape[1])
    return vtk_arr


def convert_continuous_probabilities_to_class_integer(prop_arr):
    assert prop_arr.ndim == 2, "property array is not a 2D array. Each row a vector of continuous prob"
    return np.argmax(prop_arr, axis=1)


def add_vtk_data_array_to_vtk_object(vtk_object, vtk_array):
    if type(vtk_object) == vtkStructuredGrid or \
            type(vtk_object) == vtkUnstructuredGrid or \
            type(vtk_object) == vtkPolyData:
        assert vtk_object.GetNumberOfPoints() == vtk_array.GetNumberOfTuples(), \
            "Num of Tuples is different than number of points on vtk object"
        vtk_object.GetPointData().AddArray(vtk_array)
    elif type(vtk_object) == vtkImageData:
        assert vtk_object.GetNumberOfCells() == vtk_array.GetNumberOfTuples(), \
            "Num of Tuples is different than number of cells on vtk object"
        vtk_object.GetCellData().AddArray(vtk_array)
    else:
        raise ValueError("vtk_object is not a vtkStructuredGrid | vtkImageData | vtkUnstructuredGrid")
    return vtk_object


def add_np_property_to_vtk_object(vtk_object, prop_name, prop_arr, continuous=True):
    if continuous:
        add_vtk_data_array_to_vtk_object(vtk_object, create_continuous_property_vtk_array(prop_name, prop_arr))
    else:
        add_vtk_data_array_to_vtk_object(vtk_object, create_discrete_property_vtk_array(prop_name, prop_arr))


def vertically_exaggerate_vtk_object(vtk_object, vertical_exaggeration):
    transform = vtkTransform()
    transform.Scale(1, 1, vertical_exaggeration)
    transform.Update()
    transformFilter = vtkTransformFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(vtk_object)
    transformFilter.Update()
    return transformFilter.GetOutput()


def get_resultant_bounds_from_vtk_objects(*vtk_objects, xy_buffer=0, z_buffer=0):
    """
    Given multiple vtk based datasets find the resulting bounding box for all the data. The computed bounds:
    [xmin, xmax, ymin, ymax, zmin, zmax] will be the resulting bounding box bounds plus a buffer.
    """

    assert vtk_objects, "There are no objects supplied to function get_resultant_bounds_from_vtk_objects"

    bounding_box = vtkBoundingBox()

    for vtk_object in vtk_objects:
        assert isinstance(vtk_object, vtkDataSet), 'Inputted object is supposed to be a subclass of vtkDataset,' \
                                                   ' however got a {0} instead'.format(type(vtk_object))
        bounding_box.AddBounds(vtk_object.GetBounds())

    bounds_min = bounding_box.GetMinPoint()
    bounds_max = bounding_box.GetMaxPoint()
    data_bounds = [bounds_min[0],
                   bounds_max[0],
                   bounds_min[1],
                   bounds_max[1],
                   bounds_min[2],
                   bounds_max[2]]

    # Generate Grid Pts
    dx = data_bounds[1] - data_bounds[0]
    dy = data_bounds[3] - data_bounds[2]
    dz = data_bounds[5] - data_bounds[4]
    bounds = np.zeros(6)
    bounds[0] = data_bounds[0] - xy_buffer * dx
    bounds[1] = data_bounds[1] + xy_buffer * dx
    bounds[2] = data_bounds[2] - xy_buffer * dy
    bounds[3] = data_bounds[3] + xy_buffer * dy
    bounds[4] = data_bounds[4] - z_buffer * dz
    bounds[5] = data_bounds[5] + z_buffer * dz

    return bounds


def create_vtk_polydata_from_coords_and_property(coords: np.ndarray, prop: np.ndarray, prop_name: str):
    points_vtk = vtkPoints()
    vertices = vtkCellArray()
    for pt in coords:
        pt_id = points_vtk.InsertNextPoint(pt)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(pt_id)
    poly = vtkPolyData()
    poly.SetPoints(points_vtk)
    poly.SetVerts(vertices)

    add_np_property_to_vtk_object(poly, prop_name, prop)
    return poly


def CreateLUT(min, max):
    """
    Manual creation of the "Paired" color map. It maps property values (scalars) with the specified range (min, max)
    into RGB values. Scalar values outside of this range will be set to extremal RGB values in the color look up table.
    :param min: minimum scalar value
    :param max: maximum scalar value
    :return: a vtkLookupTable for a "Paired" color map within the scalar range of (min, max)
    """
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(min, max)
    lut.Build()
    ctf = vtkColorTransferFunction()
    ctf.SetColorSpaceToRGB()

    ctf.AddRGBPoint(0.000000, 0.650980, 0.807843, 0.890196)
    ctf.AddRGBPoint(0.003922, 0.628143, 0.793295, 0.882245)
    ctf.AddRGBPoint(0.007843, 0.605306, 0.778747, 0.874295)
    ctf.AddRGBPoint(0.011765, 0.582468, 0.764198, 0.866344)
    ctf.AddRGBPoint(0.015686, 0.559631, 0.749650, 0.858393)
    ctf.AddRGBPoint(0.019608, 0.536794, 0.735102, 0.850442)
    ctf.AddRGBPoint(0.023529, 0.513956, 0.720554, 0.842491)
    ctf.AddRGBPoint(0.027451, 0.491119, 0.706005, 0.834541)
    ctf.AddRGBPoint(0.031373, 0.468281, 0.691457, 0.826590)
    ctf.AddRGBPoint(0.035294, 0.445444, 0.676909, 0.818639)
    ctf.AddRGBPoint(0.039216, 0.422607, 0.662361, 0.810688)
    ctf.AddRGBPoint(0.043137, 0.399769, 0.647812, 0.802737)
    ctf.AddRGBPoint(0.047059, 0.376932, 0.633264, 0.794787)
    ctf.AddRGBPoint(0.050980, 0.354095, 0.618716, 0.786836)
    ctf.AddRGBPoint(0.054902, 0.331257, 0.604168, 0.778885)
    ctf.AddRGBPoint(0.058824, 0.308420, 0.589619, 0.770934)
    ctf.AddRGBPoint(0.062745, 0.285582, 0.575071, 0.762983)
    ctf.AddRGBPoint(0.066667, 0.262745, 0.560523, 0.755033)
    ctf.AddRGBPoint(0.070588, 0.239908, 0.545975, 0.747082)
    ctf.AddRGBPoint(0.074510, 0.217070, 0.531426, 0.739131)
    ctf.AddRGBPoint(0.078431, 0.194233, 0.516878, 0.731180)
    ctf.AddRGBPoint(0.082353, 0.171396, 0.502330, 0.723230)
    ctf.AddRGBPoint(0.086275, 0.148558, 0.487782, 0.715279)
    ctf.AddRGBPoint(0.090196, 0.125721, 0.473233, 0.707328)
    ctf.AddRGBPoint(0.094118, 0.141915, 0.484844, 0.700069)
    ctf.AddRGBPoint(0.098039, 0.166782, 0.502268, 0.692964)
    ctf.AddRGBPoint(0.101961, 0.191649, 0.519692, 0.685859)
    ctf.AddRGBPoint(0.105882, 0.216517, 0.537116, 0.678754)
    ctf.AddRGBPoint(0.109804, 0.241384, 0.554541, 0.671649)
    ctf.AddRGBPoint(0.113725, 0.266251, 0.571965, 0.664544)
    ctf.AddRGBPoint(0.117647, 0.291119, 0.589389, 0.657439)
    ctf.AddRGBPoint(0.121569, 0.315986, 0.606813, 0.650335)
    ctf.AddRGBPoint(0.125490, 0.340854, 0.624237, 0.643230)
    ctf.AddRGBPoint(0.129412, 0.365721, 0.641661, 0.636125)
    ctf.AddRGBPoint(0.133333, 0.390588, 0.659085, 0.629020)
    ctf.AddRGBPoint(0.137255, 0.415456, 0.676509, 0.621915)
    ctf.AddRGBPoint(0.141176, 0.440323, 0.693933, 0.614810)
    ctf.AddRGBPoint(0.145098, 0.465190, 0.711357, 0.607705)
    ctf.AddRGBPoint(0.149020, 0.490058, 0.728781, 0.600600)
    ctf.AddRGBPoint(0.152941, 0.514925, 0.746205, 0.593495)
    ctf.AddRGBPoint(0.156863, 0.539792, 0.763629, 0.586390)
    ctf.AddRGBPoint(0.160784, 0.564660, 0.781053, 0.579285)
    ctf.AddRGBPoint(0.164706, 0.589527, 0.798478, 0.572180)
    ctf.AddRGBPoint(0.168627, 0.614394, 0.815902, 0.565075)
    ctf.AddRGBPoint(0.172549, 0.639262, 0.833326, 0.557970)
    ctf.AddRGBPoint(0.176471, 0.664129, 0.850750, 0.550865)
    ctf.AddRGBPoint(0.180392, 0.688997, 0.868174, 0.543760)
    ctf.AddRGBPoint(0.184314, 0.684368, 0.867728, 0.531057)
    ctf.AddRGBPoint(0.188235, 0.662884, 0.857070, 0.515156)
    ctf.AddRGBPoint(0.192157, 0.641399, 0.846413, 0.499254)
    ctf.AddRGBPoint(0.196078, 0.619915, 0.835755, 0.483353)
    ctf.AddRGBPoint(0.200000, 0.598431, 0.825098, 0.467451)
    ctf.AddRGBPoint(0.203922, 0.576947, 0.814441, 0.451549)
    ctf.AddRGBPoint(0.207843, 0.555463, 0.803783, 0.435648)
    ctf.AddRGBPoint(0.211765, 0.533979, 0.793126, 0.419746)
    ctf.AddRGBPoint(0.215686, 0.512495, 0.782468, 0.403845)
    ctf.AddRGBPoint(0.219608, 0.491011, 0.771811, 0.387943)
    ctf.AddRGBPoint(0.223529, 0.469527, 0.761153, 0.372042)
    ctf.AddRGBPoint(0.227451, 0.448043, 0.750496, 0.356140)
    ctf.AddRGBPoint(0.231373, 0.426559, 0.739839, 0.340238)
    ctf.AddRGBPoint(0.235294, 0.405075, 0.729181, 0.324337)
    ctf.AddRGBPoint(0.239216, 0.383591, 0.718524, 0.308435)
    ctf.AddRGBPoint(0.243137, 0.362107, 0.707866, 0.292534)
    ctf.AddRGBPoint(0.247059, 0.340623, 0.697209, 0.276632)
    ctf.AddRGBPoint(0.250980, 0.319139, 0.686551, 0.260730)
    ctf.AddRGBPoint(0.254902, 0.297655, 0.675894, 0.244829)
    ctf.AddRGBPoint(0.258824, 0.276171, 0.665236, 0.228927)
    ctf.AddRGBPoint(0.262745, 0.254687, 0.654579, 0.213026)
    ctf.AddRGBPoint(0.266667, 0.233203, 0.643922, 0.197124)
    ctf.AddRGBPoint(0.270588, 0.211719, 0.633264, 0.181223)
    ctf.AddRGBPoint(0.274510, 0.215379, 0.626990, 0.180930)
    ctf.AddRGBPoint(0.278431, 0.249212, 0.625975, 0.199369)
    ctf.AddRGBPoint(0.282353, 0.283045, 0.624960, 0.217809)
    ctf.AddRGBPoint(0.286275, 0.316878, 0.623945, 0.236248)
    ctf.AddRGBPoint(0.290196, 0.350711, 0.622930, 0.254687)
    ctf.AddRGBPoint(0.294118, 0.384544, 0.621915, 0.273126)
    ctf.AddRGBPoint(0.298039, 0.418378, 0.620900, 0.291565)
    ctf.AddRGBPoint(0.301961, 0.452211, 0.619885, 0.310004)
    ctf.AddRGBPoint(0.305882, 0.486044, 0.618870, 0.328443)
    ctf.AddRGBPoint(0.309804, 0.519877, 0.617855, 0.346882)
    ctf.AddRGBPoint(0.313725, 0.553710, 0.616840, 0.365321)
    ctf.AddRGBPoint(0.317647, 0.587543, 0.615825, 0.383760)
    ctf.AddRGBPoint(0.321569, 0.621376, 0.614810, 0.402199)
    ctf.AddRGBPoint(0.325490, 0.655210, 0.613795, 0.420638)
    ctf.AddRGBPoint(0.329412, 0.689043, 0.612780, 0.439077)
    ctf.AddRGBPoint(0.333333, 0.722876, 0.611765, 0.457516)
    ctf.AddRGBPoint(0.337255, 0.756709, 0.610750, 0.475955)
    ctf.AddRGBPoint(0.341176, 0.790542, 0.609735, 0.494394)
    ctf.AddRGBPoint(0.345098, 0.824375, 0.608720, 0.512834)
    ctf.AddRGBPoint(0.349020, 0.858208, 0.607705, 0.531273)
    ctf.AddRGBPoint(0.352941, 0.892042, 0.606690, 0.549712)
    ctf.AddRGBPoint(0.356863, 0.925875, 0.605675, 0.568151)
    ctf.AddRGBPoint(0.360784, 0.959708, 0.604660, 0.586590)
    ctf.AddRGBPoint(0.364706, 0.983206, 0.598016, 0.594233)
    ctf.AddRGBPoint(0.368627, 0.979146, 0.576363, 0.573087)
    ctf.AddRGBPoint(0.372549, 0.975087, 0.554710, 0.551942)
    ctf.AddRGBPoint(0.376471, 0.971027, 0.533057, 0.530796)
    ctf.AddRGBPoint(0.380392, 0.966967, 0.511403, 0.509650)
    ctf.AddRGBPoint(0.384314, 0.962907, 0.489750, 0.488504)
    ctf.AddRGBPoint(0.388235, 0.958847, 0.468097, 0.467359)
    ctf.AddRGBPoint(0.392157, 0.954787, 0.446444, 0.446213)
    ctf.AddRGBPoint(0.396078, 0.950727, 0.424790, 0.425067)
    ctf.AddRGBPoint(0.400000, 0.946667, 0.403137, 0.403922)
    ctf.AddRGBPoint(0.403922, 0.942607, 0.381484, 0.382776)
    ctf.AddRGBPoint(0.407843, 0.938547, 0.359831, 0.361630)
    ctf.AddRGBPoint(0.411765, 0.934487, 0.338178, 0.340484)
    ctf.AddRGBPoint(0.415686, 0.930427, 0.316524, 0.319339)
    ctf.AddRGBPoint(0.419608, 0.926367, 0.294871, 0.298193)
    ctf.AddRGBPoint(0.423529, 0.922307, 0.273218, 0.277047)
    ctf.AddRGBPoint(0.427451, 0.918247, 0.251565, 0.255902)
    ctf.AddRGBPoint(0.431373, 0.914187, 0.229912, 0.234756)
    ctf.AddRGBPoint(0.435294, 0.910127, 0.208258, 0.213610)
    ctf.AddRGBPoint(0.439216, 0.906067, 0.186605, 0.192464)
    ctf.AddRGBPoint(0.443137, 0.902007, 0.164952, 0.171319)
    ctf.AddRGBPoint(0.447059, 0.897947, 0.143299, 0.150173)
    ctf.AddRGBPoint(0.450980, 0.893887, 0.121646, 0.129027)
    ctf.AddRGBPoint(0.454902, 0.890596, 0.104498, 0.111080)
    ctf.AddRGBPoint(0.458824, 0.894994, 0.132411, 0.125121)
    ctf.AddRGBPoint(0.462745, 0.899393, 0.160323, 0.139162)
    ctf.AddRGBPoint(0.466667, 0.903791, 0.188235, 0.153203)
    ctf.AddRGBPoint(0.470588, 0.908189, 0.216148, 0.167243)
    ctf.AddRGBPoint(0.474510, 0.912587, 0.244060, 0.181284)
    ctf.AddRGBPoint(0.478431, 0.916986, 0.271972, 0.195325)
    ctf.AddRGBPoint(0.482353, 0.921384, 0.299885, 0.209366)
    ctf.AddRGBPoint(0.486275, 0.925782, 0.327797, 0.223406)
    ctf.AddRGBPoint(0.490196, 0.930181, 0.355709, 0.237447)
    ctf.AddRGBPoint(0.494118, 0.934579, 0.383622, 0.251488)
    ctf.AddRGBPoint(0.498039, 0.938977, 0.411534, 0.265529)
    ctf.AddRGBPoint(0.501961, 0.943376, 0.439446, 0.279569)
    ctf.AddRGBPoint(0.505882, 0.947774, 0.467359, 0.293610)
    ctf.AddRGBPoint(0.509804, 0.952172, 0.495271, 0.307651)
    ctf.AddRGBPoint(0.513725, 0.956571, 0.523183, 0.321692)
    ctf.AddRGBPoint(0.517647, 0.960969, 0.551096, 0.335732)
    ctf.AddRGBPoint(0.521569, 0.965367, 0.579008, 0.349773)
    ctf.AddRGBPoint(0.525490, 0.969765, 0.606920, 0.363814)
    ctf.AddRGBPoint(0.529412, 0.974164, 0.634833, 0.377855)
    ctf.AddRGBPoint(0.533333, 0.978562, 0.662745, 0.391895)
    ctf.AddRGBPoint(0.537255, 0.982960, 0.690657, 0.405936)
    ctf.AddRGBPoint(0.541176, 0.987359, 0.718570, 0.419977)
    ctf.AddRGBPoint(0.545098, 0.991757, 0.746482, 0.434018)
    ctf.AddRGBPoint(0.549020, 0.992464, 0.739177, 0.418224)
    ctf.AddRGBPoint(0.552941, 0.992803, 0.728351, 0.399446)
    ctf.AddRGBPoint(0.556863, 0.993141, 0.717524, 0.380669)
    ctf.AddRGBPoint(0.560784, 0.993479, 0.706697, 0.361892)
    ctf.AddRGBPoint(0.564706, 0.993818, 0.695871, 0.343114)
    ctf.AddRGBPoint(0.568627, 0.994156, 0.685044, 0.324337)
    ctf.AddRGBPoint(0.572549, 0.994494, 0.674218, 0.305559)
    ctf.AddRGBPoint(0.576471, 0.994833, 0.663391, 0.286782)
    ctf.AddRGBPoint(0.580392, 0.995171, 0.652564, 0.268005)
    ctf.AddRGBPoint(0.584314, 0.995509, 0.641738, 0.249227)
    ctf.AddRGBPoint(0.588235, 0.995848, 0.630911, 0.230450)
    ctf.AddRGBPoint(0.592157, 0.996186, 0.620085, 0.211672)
    ctf.AddRGBPoint(0.596078, 0.996524, 0.609258, 0.192895)
    ctf.AddRGBPoint(0.600000, 0.996863, 0.598431, 0.174118)
    ctf.AddRGBPoint(0.603922, 0.997201, 0.587605, 0.155340)
    ctf.AddRGBPoint(0.607843, 0.997539, 0.576778, 0.136563)
    ctf.AddRGBPoint(0.611765, 0.997878, 0.565952, 0.117785)
    ctf.AddRGBPoint(0.615686, 0.998216, 0.555125, 0.099008)
    ctf.AddRGBPoint(0.619608, 0.998554, 0.544298, 0.080231)
    ctf.AddRGBPoint(0.623529, 0.998893, 0.533472, 0.061453)
    ctf.AddRGBPoint(0.627451, 0.999231, 0.522645, 0.042676)
    ctf.AddRGBPoint(0.631373, 0.999569, 0.511819, 0.023899)
    ctf.AddRGBPoint(0.635294, 0.999908, 0.500992, 0.005121)
    ctf.AddRGBPoint(0.639216, 0.993479, 0.504314, 0.026328)
    ctf.AddRGBPoint(0.643137, 0.984514, 0.512941, 0.062530)
    ctf.AddRGBPoint(0.647059, 0.975548, 0.521569, 0.098731)
    ctf.AddRGBPoint(0.650980, 0.966582, 0.530196, 0.134933)
    ctf.AddRGBPoint(0.654902, 0.957616, 0.538824, 0.171134)
    ctf.AddRGBPoint(0.658824, 0.948651, 0.547451, 0.207336)
    ctf.AddRGBPoint(0.662745, 0.939685, 0.556078, 0.243537)
    ctf.AddRGBPoint(0.666667, 0.930719, 0.564706, 0.279739)
    ctf.AddRGBPoint(0.670588, 0.921753, 0.573333, 0.315940)
    ctf.AddRGBPoint(0.674510, 0.912787, 0.581961, 0.352141)
    ctf.AddRGBPoint(0.678431, 0.903822, 0.590588, 0.388343)
    ctf.AddRGBPoint(0.682353, 0.894856, 0.599216, 0.424544)
    ctf.AddRGBPoint(0.686275, 0.885890, 0.607843, 0.460746)
    ctf.AddRGBPoint(0.690196, 0.876924, 0.616471, 0.496947)
    ctf.AddRGBPoint(0.694118, 0.867958, 0.625098, 0.533149)
    ctf.AddRGBPoint(0.698039, 0.858993, 0.633726, 0.569350)
    ctf.AddRGBPoint(0.701961, 0.850027, 0.642353, 0.605552)
    ctf.AddRGBPoint(0.705882, 0.841061, 0.650980, 0.641753)
    ctf.AddRGBPoint(0.709804, 0.832095, 0.659608, 0.677955)
    ctf.AddRGBPoint(0.713725, 0.823130, 0.668235, 0.714156)
    ctf.AddRGBPoint(0.717647, 0.814164, 0.676863, 0.750358)
    ctf.AddRGBPoint(0.721569, 0.805198, 0.685490, 0.786559)
    ctf.AddRGBPoint(0.725490, 0.796232, 0.694118, 0.822760)
    ctf.AddRGBPoint(0.729412, 0.783299, 0.687243, 0.833679)
    ctf.AddRGBPoint(0.733333, 0.767059, 0.667451, 0.823529)
    ctf.AddRGBPoint(0.737255, 0.750819, 0.647659, 0.813379)
    ctf.AddRGBPoint(0.741176, 0.734579, 0.627866, 0.803230)
    ctf.AddRGBPoint(0.745098, 0.718339, 0.608074, 0.793080)
    ctf.AddRGBPoint(0.749020, 0.702099, 0.588281, 0.782930)
    ctf.AddRGBPoint(0.752941, 0.685859, 0.568489, 0.772780)
    ctf.AddRGBPoint(0.756863, 0.669619, 0.548697, 0.762630)
    ctf.AddRGBPoint(0.760784, 0.653379, 0.528904, 0.752480)
    ctf.AddRGBPoint(0.764706, 0.637140, 0.509112, 0.742330)
    ctf.AddRGBPoint(0.768627, 0.620900, 0.489320, 0.732180)
    ctf.AddRGBPoint(0.772549, 0.604660, 0.469527, 0.722030)
    ctf.AddRGBPoint(0.776471, 0.588420, 0.449735, 0.711880)
    ctf.AddRGBPoint(0.780392, 0.572180, 0.429942, 0.701730)
    ctf.AddRGBPoint(0.784314, 0.555940, 0.410150, 0.691580)
    ctf.AddRGBPoint(0.788235, 0.539700, 0.390358, 0.681430)
    ctf.AddRGBPoint(0.792157, 0.523460, 0.370565, 0.671280)
    ctf.AddRGBPoint(0.796078, 0.507220, 0.350773, 0.661130)
    ctf.AddRGBPoint(0.800000, 0.490980, 0.330980, 0.650980)
    ctf.AddRGBPoint(0.803922, 0.474740, 0.311188, 0.640830)
    ctf.AddRGBPoint(0.807843, 0.458501, 0.291396, 0.630681)
    ctf.AddRGBPoint(0.811765, 0.442261, 0.271603, 0.620531)
    ctf.AddRGBPoint(0.815686, 0.426021, 0.251811, 0.610381)
    ctf.AddRGBPoint(0.819608, 0.424852, 0.251150, 0.603860)
    ctf.AddRGBPoint(0.823529, 0.450058, 0.283968, 0.603691)
    ctf.AddRGBPoint(0.827451, 0.475263, 0.316786, 0.603522)
    ctf.AddRGBPoint(0.831373, 0.500469, 0.349604, 0.603353)
    ctf.AddRGBPoint(0.835294, 0.525675, 0.382422, 0.603183)
    ctf.AddRGBPoint(0.839216, 0.550880, 0.415240, 0.603014)
    ctf.AddRGBPoint(0.843137, 0.576086, 0.448058, 0.602845)
    ctf.AddRGBPoint(0.847059, 0.601292, 0.480877, 0.602676)
    ctf.AddRGBPoint(0.850980, 0.626498, 0.513695, 0.602507)
    ctf.AddRGBPoint(0.854902, 0.651703, 0.546513, 0.602338)
    ctf.AddRGBPoint(0.858824, 0.676909, 0.579331, 0.602168)
    ctf.AddRGBPoint(0.862745, 0.702115, 0.612149, 0.601999)
    ctf.AddRGBPoint(0.866667, 0.727320, 0.644967, 0.601830)
    ctf.AddRGBPoint(0.870588, 0.752526, 0.677785, 0.601661)
    ctf.AddRGBPoint(0.874510, 0.777732, 0.710604, 0.601492)
    ctf.AddRGBPoint(0.878431, 0.802937, 0.743422, 0.601323)
    ctf.AddRGBPoint(0.882353, 0.828143, 0.776240, 0.601153)
    ctf.AddRGBPoint(0.886275, 0.853349, 0.809058, 0.600984)
    ctf.AddRGBPoint(0.890196, 0.878554, 0.841876, 0.600815)
    ctf.AddRGBPoint(0.894118, 0.903760, 0.874694, 0.600646)
    ctf.AddRGBPoint(0.898039, 0.928966, 0.907512, 0.600477)
    ctf.AddRGBPoint(0.901961, 0.954171, 0.940331, 0.600308)
    ctf.AddRGBPoint(0.905882, 0.979377, 0.973149, 0.600138)
    ctf.AddRGBPoint(0.909804, 0.997601, 0.994894, 0.596524)
    ctf.AddRGBPoint(0.913725, 0.984406, 0.966813, 0.577409)
    ctf.AddRGBPoint(0.917647, 0.971211, 0.938731, 0.558293)
    ctf.AddRGBPoint(0.921569, 0.958016, 0.910650, 0.539177)
    ctf.AddRGBPoint(0.925490, 0.944821, 0.882568, 0.520062)
    ctf.AddRGBPoint(0.929412, 0.931626, 0.854487, 0.500946)
    ctf.AddRGBPoint(0.933333, 0.918431, 0.826405, 0.481830)
    ctf.AddRGBPoint(0.937255, 0.905236, 0.798324, 0.462714)
    ctf.AddRGBPoint(0.941176, 0.892042, 0.770242, 0.443599)
    ctf.AddRGBPoint(0.945098, 0.878847, 0.742161, 0.424483)
    ctf.AddRGBPoint(0.949020, 0.865652, 0.714079, 0.405367)
    ctf.AddRGBPoint(0.952941, 0.852457, 0.685998, 0.386251)
    ctf.AddRGBPoint(0.956863, 0.839262, 0.657916, 0.367136)
    ctf.AddRGBPoint(0.960784, 0.826067, 0.629835, 0.348020)
    ctf.AddRGBPoint(0.964706, 0.812872, 0.601753, 0.328904)
    ctf.AddRGBPoint(0.968627, 0.799677, 0.573672, 0.309789)
    ctf.AddRGBPoint(0.972549, 0.786482, 0.545590, 0.290673)
    ctf.AddRGBPoint(0.976471, 0.773287, 0.517509, 0.271557)
    ctf.AddRGBPoint(0.980392, 0.760092, 0.489427, 0.252441)
    ctf.AddRGBPoint(0.984314, 0.746897, 0.461346, 0.233326)
    ctf.AddRGBPoint(0.988235, 0.733702, 0.433264, 0.214210)
    ctf.AddRGBPoint(0.992157, 0.720508, 0.405183, 0.195094)
    ctf.AddRGBPoint(0.996078, 0.707313, 0.377101, 0.175978)
    ctf.AddRGBPoint(1.000000, 0.694118, 0.349020, 0.156863)

    for i in range(256):
        r, g, b = ctf.GetColor(float(i / 255.0))
        lut.SetTableValue(i, r, g, b)

    return lut


def write_slices_from_unstructured_grid_as_pictures(grid: vtkUnstructuredGrid, output_dir):
    """
    This function will obtain 9 slices of the vtkUnstructuredGrid (3 per axis (x, y, z) - left, middle, right) to
    visualize the 1-component scalar array (vtkDoubleArray) it's attributed with. The slices will be converted into
    images using a specified color map.
    In order to achieve this objective a slice represented by a vtkImageData is used to probe the vtkUnstructuredGrid
    where the scalar property will be interpolated from.
    For non-z-slices vtkImageData will have to be re-orientated to create a proper XY image from. vtkImagePermute class
    is used for this purpose.
    :param grid: vtkUnstructuredGrid containing a 1-component scalar array (vtkDoubleArray)
    :param output_dir: output directory where png images will be written to.
    :return: nothing
    """

    # resolution of outputted images: pixel's spatial resolution
    spacing = 0.5

    # get the value range of the scalar data
    value_array = grid.GetPointData().GetArray("Scalar Field")
    vmin, vmax = value_array.GetRange()

    # create "Paired" color map with the given range
    lut = CreateLUT(vmin, vmax)
    xmin, xmax, ymin, ymax, zmin, zmax = grid.GetBounds()

    #  specify x/y/z positions of slices
    #  3 per direction : {left, middle, right}
    slice_pos = []
    # positions of x-slices
    slice_pos.append(xmin)
    slice_pos.append(xmin + ((xmax - xmin) / 2.0))
    slice_pos.append(xmax)

    # positions of y-slices
    slice_pos.append(ymin)
    slice_pos.append(ymin + ((ymax - ymin) / 2.0))
    slice_pos.append(ymax)

    # positions of z-slices
    slice_pos.append(zmin)
    slice_pos.append(zmin + ((zmax - zmin) / 2.0))
    slice_pos.append(zmax)

    for i, pos in enumerate(slice_pos):
        slice = vtkImageData()
        slice.SetSpacing(spacing, spacing, spacing)
        # x-slices
        if i < 3:
            tag = "x-slice_" + str(i)
            s_xmin = pos
            s_xmax = pos
            s_ymin = ymin
            s_ymax = ymax
            s_zmin = zmin
            s_zmax = zmax
        # y-slices
        if 2 < i < 6:
            tag = "y-slice_" + str(i)
            s_xmin = xmin
            s_xmax = xmax
            s_ymin = pos
            s_ymax = pos
            s_zmin = zmin
            s_zmax = zmax
        # z-slices
        if i > 5:
            tag = "z-slice_" + str(i)
            s_xmin = xmin
            s_xmax = xmax
            s_ymin = ymin
            s_ymax = ymax
            s_zmin = pos
            s_zmax = pos
        dim_x = int(math.ceil((s_xmax - s_xmin) / spacing))
        dim_y = int(math.ceil((s_ymax - s_ymin) / spacing))
        dim_z = int(math.ceil((s_zmax - s_zmin) / spacing))

        # x-slices
        if i < 3:
            extent_x = 0
            extent_y = dim_y - 1
            extent_z = dim_z - 1
        # y-slices
        if 2 < i < 6:
            extent_x = dim_x - 1
            extent_y = 0
            extent_z = dim_z - 1
        # z-slices
        if i > 5:
            extent_x = dim_x - 1
            extent_y = dim_y - 1
            extent_z = 0
        slice.SetExtent(0, extent_x, 0, extent_y, 0, extent_z)
        slice.SetOrigin(s_xmin, s_ymin, s_zmin)

        probe = vtkProbeFilter()
        probe.SetInputData(slice)
        probe.SetSourceData(grid)
        probe.Update()

        out = probe.GetOutput()
        arr = out.GetPointData().GetArray("Scalar Field")
        # set scalars to this array - needed for vtkImageMapToColors filter
        # needs to know which scalar array to use to map scalars to colors
        out.GetPointData().SetScalars(arr)

        # probe_writer = vtkXMLImageDataWriter()
        # img_filename = "D:/image_slice_probe_python_" + str(i) + ".vti"
        # probe_writer.SetFileName(img_filename)
        # probe_writer.SetInputData(out)
        # probe_writer.Write()

        if i < 6:
            # for x and y slices images orientations have to be switched so that the png writer
            # will correctly export the image - images must have x and y orientation
            # z slices are already oriented in x and y
            permute = vtkImagePermute()
            permute.SetInputData(out)
            # x-slice
            if i < 3:
                # z direction will become x direction
                permute.SetFilteredAxes(2, 1, 0)
            # y-slice
            if i > 2:
                # z direction will become y direction
                permute.SetFilteredAxes(0, 2, 1)
            permute.Update()

            permute_img = permute.GetOutput()
            perm_arr = permute_img.GetPointData().GetArray(0)
            # set scalars to this array - needed for vtkImageMapToColors filter
            # needs to know which scalar array to use to map scalars to colors
            permute_img.GetPointData().SetScalars(perm_arr)

            # perm_writer = vtkXMLImageDataWriter()
            # perm_filename = "D:/image_slice_permute_python_" + str(i) + ".vti"
            # perm_writer.SetFileName(perm_filename)
            # perm_writer.SetInputData(permute_img)
            # perm_writer.Write()

            scalarValuesToColors = vtkImageMapToColors()
            scalarValuesToColors.SetLookupTable(lut)
            scalarValuesToColors.SetOutputFormatToRGB()
            scalarValuesToColors.SetInputData(permute_img)
            scalarValuesToColors.Update()
        else:
            scalarValuesToColors = vtkImageMapToColors()
            scalarValuesToColors.SetLookupTable(lut)
            scalarValuesToColors.SetOutputFormatToRGB()
            scalarValuesToColors.SetInputData(out)
            scalarValuesToColors.Update()

        png_writer = vtkPNGWriter()
        png_filename = output_dir + "/image_slice_" + str(i) + ".png"
        png_writer.SetFileName(png_filename)
        png_writer.SetInputData(scalarValuesToColors.GetOutput())
        png_writer.Write()
