import os
import vtkmodules.all as vtk


def reader_xml_polydata_file(pd_filename: str):
    """
    :param pd_filename: an XML vtkPolyData file format (*.vtp)
    :return: vtkPolyData data structure - also carries any property data attributed to points, cells, as well as field
             data (meta data)
    """
    if not os.path.isfile(pd_filename):
        raise ValueError('File does not exist')

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(pd_filename)
    reader.Update()

    return reader.GetOutput()


def reader_unstructured_mesh_file(mesh_filename: str):
    '''
    :param mesh_filename: vtk legacy file format representing a vtkUnstructuredGrid
    :return: a vtkUnstructuredGrid (just geometry and topology - no property)
    '''
    if not os.path.isfile(mesh_filename):
        raise ValueError('File does not exist')

    filename, file_extension = os.path.splitext(mesh_filename)
    if file_extension == '.vtk':
        reader = vtk.vtkUnstructuredGridReader()
    else:
        reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(mesh_filename)
    reader.Update()

    return reader.GetOutput()
