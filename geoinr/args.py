import os
import platform
import torch
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_pretrained_model_file(init_geo: str, args: argparse.Namespace):
    assert init_geo in ['plane', 'sphere']
    pretrained_filename = 'model/mlp/pretrained/' + init_geo

    # add activation
    pretrained_filename += '_' + args.activation

    # add NN model parameters
    pretrained_filename += "_model" + str(args.num_hidden_layers) + "L_"
    pretrained_filename += str(args.embed_dim) + '_'

    if args.activation == 'splus':
        pretrained_filename += str(int(args.beta)) + 'b'

    pretrained_filename += '.pth'

    return os.path.join(ROOT_DIR, pretrained_filename)


def valid_range(nmin, nmax):
    '''
    Checks if an numerical argument's value is within the allowable range
    :param nmin: smallest allowable value for argument
    :param nmax: largest allowable value for agrument
    :return: nothing. BUT if argument value is outside of allowable range throw exception
    '''

    class ValidRange(argparse.Action):
        def __call__(self, parser, args, value, option_string=None):
            if not nmin <= value <= nmax:
                msg = 'argument "{f}={value}" outside of allowed range [{nmin},{nmax}]'.format(f=self.dest, value=value,
                                                                                               nmin=nmin, nmax=nmax)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, value)

    return ValidRange


def valid_scaler_approach(scaler_type: str):
    if scaler_type not in ["minmax", "standard", "custom1", "custom2", "custom3"]:
        msg = f'argument {scaler_type} is not in the implemented types [minmax, standard, custom1, custom2, custom3'
        raise argparse.ArgumentTypeError(msg)
    else:
        return scaler_type


def valid_type_of_approach(approach_type: str):
    if approach_type not in ["mlp", "pe_mlp"]:
        msg = f'argument {approach_type} is not in the implemented types [minmax, standard, custom1, custom2, custom3'
        raise argparse.ArgumentTypeError(msg)
    else:
        return approach_type


def valid_filename(filename: str):
    if filename is None:
        return filename
    if filename == 'None':
        return None
    file_name, file_extension = os.path.splitext(filename)
    if file_extension == ".vtp":
        return filename
    else:
        msg = f'File extension {file_extension} in supplied filename {filename} is not a .vtp extension'
        raise argparse.ArgumentTypeError(msg)


def valid_metadata_filename(filename: str):
    if filename is None:
        return filename
    if filename == 'None':
        return None
    file_name, file_extension = os.path.splitext(filename)
    if file_extension == ".csv":
        return filename
    else:
        msg = f'File extension {file_extension} in supplied filename {filename} is not a .csv extension'
        raise argparse.ArgumentTypeError(msg)


def valid_activation_func(activation: str):
    if activation not in ['relu', 'splus', 'elu', 'gelu', 'mish', 'silu', 'selu', 'prelu', 'sin']:
        msg = f'argument {activation} is not in the implemented types [relu, elu, gelu, mish, silu, selu, prelu, sin]'
        raise argparse.ArgumentTypeError(msg)
    else:
        return activation


def arg_parse():
    '''
    Parses command line arguments
    :return:
    '''
    parser = argparse.ArgumentParser(description='Parsing arguments for GeoINR algorithm')
    # Start with the required arguments (positional)
    parser.add_argument("--dataset",
                        help="Dataset that is used for algorithm. A directory inside the datadir where dataset's data "
                             "files exist.")
    parser.add_argument("--interface_file", help="Name of interface file to use as input",
                        type=valid_filename)
    parser.add_argument("--unit_file", help="Name of unit file to use as input",
                        type=valid_filename)
    parser.add_argument("--normal_file", help="Name of normal file to use as input",
                        type=valid_filename)
    parser.add_argument("--metadata_file", help="Metadata for stratigraphy (horizon names, sequence, relationships",
                        type=valid_metadata_filename)
    parser.add_argument("--dist", help="Is distributed computing going to be used?", type=bool)
    parser.add_argument("--activation", help="Non-Linear activation function for neural networks. Accepted funcs"
                                             " are [relu, splus, elu, gelu, mish, silu, selu, prelu, sin]",
                        type=valid_activation_func)
    parser.add_argument("--beta", help="Parameter for Softplus activation function", type=float)
    parser.add_argument("--num_hidden_layers", help="Number of hidden layers in NN module", type=int)
    parser.add_argument("--embed_dim",
                        help="Embedding dimension of node's representations for all hidden layers in NN module",
                        type=int)
    # optional arguments
    parser.add_argument("--datadir", help="Directory where data exists")
    parser.add_argument("--technique", help="Type of approach used: [mlp, pe_mlp]", type=valid_type_of_approach)
    parser.add_argument("--lambda_g", help="Lambda parameter for weight global constraint", type=float)
    parser.add_argument("--learning_rate", help="Learning rate for optimizer", type=float, action=valid_range(0.0, 1.0))
    parser.add_argument("--weight_decay", help="Weight decay for weight matrices L2 norm",
                        type=float,
                        action=valid_range(0.0, 1.0))
    parser.add_argument("--num_epocs", help="Number of epocs for training", type=int)
    parser.add_argument("--batch_size", help="Number of training nodes in training batch", type=int)
    parser.add_argument("--dropout", help="Probability of dropout of neurons during training", type=float,
                        action=valid_range(0.0, 1.0))
    parser.add_argument("--scale_method", help="Scale method for input features: [minMax, standard, "
                                               "custom1, custom2, custom3]", type=valid_scaler_approach)
    parser.add_argument("--scale_range", nargs=2, metavar=('a', 'b'), help="Range for scaled coords --scale_range"
                                                                           " -1.0 1.0", type=float)
    parser.add_argument("--mse", help="MeanSquaredError Mode. [mean, sum]")
    parser.add_argument("--xy_resolution", help="X/Y resolution of an evaluation grid", type=float)
    parser.add_argument("--z_resolution", help="Z/Depth resolution of an evaluation grid", type=float)
    parser.add_argument("--xy_buffer", help="Resulting grid bounds XY will be data bounds + this buffer percent on"
                                            " either side", type=float)
    parser.add_argument("--z_buffer", help="Resulting grid bounds Z will be data bounds + this buffer percent on"
                                           " either side", type=float)
    parser.add_argument("--concat", help="Basic skip connections via concatenation", action='store_true')
    parser.add_argument("--nskips", help="Number of skip connection blocks", type=int)
    parser.add_argument("--omega0", help="Scaling terms for periodic sine activations (1st layer)", type=float)
    parser.add_argument("--model_dir", metavar='path', help="Location of Model for post processing")
    parser.add_argument("--debug", help="Output individual scalar field isosurfaces, uncut_isosurfaces",
                        action='store_true')
    parser.add_argument("--output_grid", help="Output volumetric grid",
                        action='store_true')
    parser.add_argument("--distance_metrics", help="Output distance metrics b/t points and isosurfaces",
                        action='store_true')
    parser.add_argument("--v_exagg", help="Vertical Exaggeration on outputted 3D geometry", type=float)
    parser.add_argument("--ngpus", help="Number of gpus to use for distributed training."
                                        " If -1, means use all available gpus", type=int)
    parser.add_argument("--n_grid_samples", help="Number of samples to take from grid for global evaluation", type=int)
    parser.add_argument("--kfold", help="Number of folds for k-fold cross validation", type=int)
    parser.add_argument("--efficient", help="Use efficient implementation of stratigraphic relations", action='store_true')
    parser.add_argument("--youngest_unit_sampled", help="If dataset has the youngest unit sampled (used for processing)", action='store_true')
    parser.set_defaults(dataset='cbaf',
                        interface_file='markers_6.vtp',
                        unit_file='intraformational.vtp',
                        normal_file=None,
                        metadata_file='marker_info.csv',
                        dist=False,
                        activation='splus',
                        beta=20,
                        num_hidden_layers=3,
                        embed_dim=256,
                        datadir='data',
                        technique='mlp',
                        lambda_g=0.0,
                        learning_rate=0.0001,
                        weight_decay=0.0,
                        grad_clip=0.0,
                        num_epocs=2000,
                        batch_size=5000,
                        dropout=0.0,
                        scale_method='custom1',
                        scale_range=[-1, 1],
                        mse='mean',
                        xy_resolution=100,
                        z_resolution=10,
                        xy_buffer=1e-5,
                        z_buffer=13,
                        concat=False,
                        nskips=2,
                        omega0=1.0,
                        model_dir=None,
                        root_dir=ROOT_DIR,
                        debug=True,
                        output_grid=True,
                        distance_metrics=False,
                        v_exagg=1,
                        ngpus=-1,
                        kfold=0,
                        n_grid_samples=5000
                        )
    params = parser.parse_args()
    params.platform = platform.system()
    params.cuda = torch.version.cuda
    return params
