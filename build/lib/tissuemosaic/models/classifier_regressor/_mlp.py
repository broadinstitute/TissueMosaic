from typing import List
import torch


def make_mlp_torch(
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        hidden_activation: torch.nn.Module,
        output_activation: torch.nn.Module) -> torch.nn.Module:
    """ Creates a multi-layer perceptron. Returns a torch.nn.module """
    # assertion
    assert isinstance(input_dim, int) and input_dim >= 1, \
        "Error. Input_dim = {0} must be int >= 1.".format(input_dim)
    assert isinstance(output_dim, int) and output_dim >= 1, \
        "Error. Output_dim = {0} must be int >= 1.".format(output_dim)
    assert hidden_dims is None or isinstance(hidden_dims, List), \
        "Error. hidden_dims must a None or a List of int (possibly empty). Received {0}".format(hidden_dims)

    # architecture
    if hidden_dims is None or len(hidden_dims) == 0:
        net = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=output_dim, bias=True),
            output_activation)
    else:
        modules = []
        tmp_dims = [input_dim] + hidden_dims

        # zip of 2 lists of different lengths terminates when shorter list terminates
        for dim_in, dim_out in zip(tmp_dims, tmp_dims[1:]):
            modules.append(torch.nn.Linear(in_features=dim_in, out_features=dim_out, bias=True))
            modules.append(hidden_activation)
        modules.append(torch.nn.Linear(in_features=tmp_dims[-1], out_features=output_dim, bias=True))
        modules.append(output_activation)
        net = torch.nn.Sequential(*modules)
    return net
