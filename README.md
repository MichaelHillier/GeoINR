# GeoINR
Geological Implicit Neural Representation for three-dimensional structural geological modelling applications.
Research carried out at the Geological Survey of Canada (Ottawa, Quebec City) and RWTH Aachen University (Germany) by Michael Hillier, J. Florian Wellmann, Eric de Kemp, Boyan Brodaric, Ernst Schetselaar, and Karine Bedard.

# Installation
Suggest using the following installation procedure using conda (https://www.anaconda.com/products/distribution)
```
conda create --name geoinr python=3.8
conda activate geoinr
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia   (note: cuda version depends on your GPU. See https://pytorch.org/get-started/locally/)
conda install -c anaconda tensorboard
conda install -c conda-forge pyvista
conda install matplotlib
conda install scikit-learn
conda install pandas
conda install seaborn -c conda-forge
```

# Inputs
Constraints are required to be in a VTK file format (good for visualization within Paraview)
Current supported constraints are interface, normal, tangent, and geological unit.

Interface Properties: 'level' describes the sequential position in which the interface was formed. The specific value is not important but the sequence is. There is support for 3 options: (3) already in the correct order for code and is normalized, (2) younger-> older : smaller value-> larger value, (1) younger->older : larger value -> smaller value. For example, consider mode (2) (1.0, 2.0, 5.0, 6.0) Level = 1.0 is the youngest interface, level = 6.0 is the oldest.
Normal Properties: 'normal' describes a 3D vector describing normal orientation of a plane.
Tangent (to be implemented): 'tangent' describes a 3D vector describing linear orientation (lineations, plunge directions, etc)
Geological Unit : 'unit' an integer describing the discrete unit or class 

# Geological Features
Currently supported geological features include conformable stratigraphy as well as unconformities. If unconformities are to be modelled, a marker info file is required for input (csv file - formation/interface name, fm_code, relationship). Furthermore, the unconformities markers/interface points must be included in the interface constraint file. The formation code (fm_code) must match the level property - they are the same thing. For example fm_code=51, and level=51.
Faults are unfortunately not supported yet in the methodology. This aspect is apart of future research.

# Technical Features
Distributed training over multiple GPUs will be released at a later date. This code will be able to be run on AWS.

# GMD Paper
This scripts used for modelling were: strat_rel_w_global.py and strat_rel_w_units_global.py.
If scripts are to be ran in the terminal the location of the GeoINR code is need to avoid module not found errors. 
Examples. On linux:
```
export PYTHONPATH="${PYTHONPATH}:/home/ubuntu/GeoINR"
```
On windows, environment variables are required to be edited. Add Variable = PYTHONPATH, Value = {PATH_TO_GeoINR}.
Then you can navigate to where the training scripts are and run (e.g., python strat_rel_w_global.py).

Geometrical initialization: To initialize network variables for planar geometry we trained using the multilayered dataset (geoinr/data/multilayer) along with the script level_constrained.py

Results were generated on Windows 10 within Pycharm using:

- pytorch 1.10.2 with cuda version 10.2
- tensorboard 2.2.1
- pyvista 0.33.2
- matplotlib 3.5.1
- scikit-learn 1.0.2
- pandas 1.4.0

# GMD case study script commands
Case Study 1
```
python strat_rel_w_global.py --dataset=unconformities --interface_file=markers.vtp --metadata_file=marker_info.csv --beta=100 --lambda_g=0.1 --num_epocs=5000 --scale_method=custom3 --xy_resolution=5000 --z_resolution=20 --xy_buffer=0.0 --z_buffer=0.0
python strat_rel_w_units_global.py --dataset=unconformities --interface_file=markers.vtp --unit_file=units.vtp --metadata_file=marker_info.csv --beta=100 --lambda_g=0.1 --num_epocs=5000 --scale_method=custom3 --xy_resolution=5000 --z_resolution=20 --xy_buffer=0.0 --z_buffer=0.0
```
Case Study 2
```
python strat_rel_w_units_global.py --dataset=cbaf --interface_file=markers_6.vtp --unit_file=intraformational.vtp --metadata_file=marker_info.csv --beta=20 --lambda_g=0.0 --num_epocs=2000 --scale_method=custom1 --xy_resolution=100 --z_resolution=10 --xy_buffer=0.0001 --z_buffer=13 --efficient --youngest_unit_sampled
python strat_units_normals.py --dataset=cbaf --interface_file=markers_6.vtp --unit_file=intraformational.vtp --normal_file=normals2.vtp --metadata_file=marker_info.csv --beta=20 --lambda_g=0.0 --num_epocs=2000 --scale_method=custom1 --xy_resolution=100 --z_resolution=10 --xy_buffer=0.0001 --z_buffer=13 --efficient --youngest_unit_sampled
python strat_units_normals.py --dataset=cbaf --interface_file=markers_16.vtp --unit_file=intraformational.vtp --normal_file=normals2.vtp --metadata_file=marker_info.csv --beta=20 --lambda_g=0.0 --num_epocs=2000 --scale_method=custom1 --xy_resolution=100 --z_resolution=10 --xy_buffer=0.0001 --z_buffer=13 --efficient --youngest_unit_sampled
python strat_units_normals.py --dataset=cbaf --interface_file=markers_26.vtp --unit_file=intraformational.vtp --normal_file=normals2.vtp --metadata_file=marker_info.csv --beta=20 --lambda_g=0.0 --num_epocs=2000 --scale_method=custom1 --xy_resolution=100 --z_resolution=10 --xy_buffer=0.0001 --z_buffer=13 --efficient --youngest_unit_sampled
python strat_units_normals.py --dataset=cbaf --interface_file=markers_45.vtp --unit_file=intraformational.vtp --normal_file=normals2.vtp --metadata_file=marker_info.csv --beta=20 --lambda_g=0.0 --num_epocs=2000 --scale_method=custom1 --xy_resolution=100 --z_resolution=10 --xy_buffer=0.0001 --z_buffer=13 --efficient --youngest_unit_sampled
```
# Other scripts
Other scripts contained within train/mlp/implicit were earlier experiments that lead to improving the methodology. Left these here because they may be useful in the future.