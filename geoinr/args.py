import os
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def arg_parse():
    '''
    Parses command line arguments
    :return:
    '''
    parser = argparse.ArgumentParser(description='Parsing arguments for Topological GNN algorithm')
    # Start with the required arguments (positional)
    parser.add_argument("--dataset",
                        help="Dataset that is used for algorithm. A directory inside the datadir where dataset's data "
                             "files exist.")
    parser.add_argument("--activation", help="Non-Linear activation function for neural networks")
    parser.add_argument("--beta", help="Parameter for Softplus activation function", type=float)
    parser.add_argument("--num_hidden_layers", help="Number of hidden layers in GNN module", type=int)
    parser.add_argument("--embed_dim",
                        help="Embedding dimension of node's representations for all hidden layers in GNN",
                        type=int)
    # optional arguments
    parser.add_argument("--datadir", help="Directory where data exists")
    parser.add_argument("--cuda", help="Flag to use GPU", type=bool)
    parser.add_argument("--technique", help="Type of approach used for GNN: [sparse_adj_matrix, pytorch_geometric, mlp]")
    parser.add_argument("--lambda_g", help="Lambda parameter for weight global constraint", type=float)
    parser.add_argument("--learning_rate", help="Learning rate for optimizer", type=float, action=valid_range(0.0, 1.0))
    parser.add_argument("--weight_decay", help="Weight decay for weight matrices L2 norm",
                        type=float,
                        action=valid_range(0.0, 1.0))
    parser.add_argument("--num_epocs", help="Number of epocs for training", type=int)
    parser.add_argument("--batch_size", help="Number of training nodes in training batch", type=int)
    parser.add_argument("--scale_isometric", help="Scale input feature isometrically if true otherwise use sci-kit "
                                                  "learn algo", type=bool)
    parser.add_argument("--rock_units", help="Creating rock unit model", type=bool)
    parser.add_argument("--mse", help="MeanSquaredError Mode. [mean, sum]")
    parser.add_argument("--xy_resolution", help="X/Y resolution of an evaluation grid", type=float)
    parser.add_argument("--z_resolution", help="Z/Depth resolution of an evaluation grid", type=float)
    parser.add_argument("--xy_buffer", help="Resulting grid bounds XY will be data bounds + this buffer percent on"
                                            " either side", type=float)
    parser.add_argument("--z_buffer", help="Resulting grid bounds Z will be data bounds + this buffer percent on"
                                            " either side")
    parser.add_argument("--concat", help="Basic skip connections via concatenation", type=bool)
    parser.add_argument("--nskips", help="Number of skip connection blocks", type=int)
    parser.add_argument("--omega0", help="Scaling terms for periodic sine activations (1st layer)", type=float)
    parser.add_argument("--model_dir", metavar='path', help="Location of Model for post processing")
    parser.set_defaults(dataset='unconformities',
                        activation='softplus',
                        beta=100,
                        num_hidden_layers=3,
                        embed_dim=256,
                        datadir='data',
                        cuda=True,
                        technique='mlp',
                        lambda_g=0.1,
                        learning_rate=0.0001,
                        weight_decay=0.0,
                        grad_clip=0.0,
                        num_epocs=5000,
                        batch_size=5000,
                        dropout=0.0,
                        scale_isometric=False,
                        rock_units=False,
                        mtl=False,
                        global_anisotropy=False,
                        mse='mean',
                        xy_resolution=5000,
                        z_resolution=20,
                        xy_buffer=0.0,
                        z_buffer=0.0,
                        concat=False,
                        nskips=2,
                        omega0=1.0,
                        model_dir=None,
                        root_dir=ROOT_DIR
                        )
    return parser.parse_args()
