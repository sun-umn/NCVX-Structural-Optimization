import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fun


def set_seed(seed):
    """
    Function to set the seed for the run
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_seeded_random_variable(latent_size, seed):
    """
    Want to try an experiment that could help the network
    generalize a bit better. Right now the best designs
    depend on the random seed that is being initialized
    """
    set_seed(seed)
    return torch.normal(mean=0.0, std=1.0, size=(3, latent_size))


# Create a custom global normalization layer for pytorch
class GlobalNormalization(nn.Module):
    """
    Class that computes the global normalization that we
    saw in the structural optimization code
    """

    def __init__(self, epsilon=1e-6):  # noqa
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):  # noqa
        var, mean = torch.var_mean(x, unbiased=False)
        net = x
        net = net - mean
        net = net * torch.rsqrt(var + self.epsilon)
        return net


# Create a layer to add offsets
class AddOffset(nn.Module):
    """
    Class that adds the weights / bias offsets & is
    trainable for the structural optimization code
    """

    def __init__(self, conv_channels, height, width, scale=10.0):  # noqa
        super().__init__()
        self.scale = torch.tensor(scale, requires_grad=True)
        self.conv_channels = conv_channels
        self.height = height
        self.width = width
        self.bias = nn.Parameter(
            torch.zeros(1, self.conv_channels, self.height, self.width),
            requires_grad=True,
        )

    def forward(self, x):  # noqa
        return x + (self.scale * self.bias)


class PixelModel(nn.Module):
    """
    Class that implements an extremely simple model that only
    utilizes the design domain for optimization instead of a DIP
    """

    def __init__(  # noqa
        self,
        args,
    ):
        super().__init__()

        # Get the height and the width
        height = args['nely']
        width = args['nelx']

        self.design_domain = nn.Parameter(torch.randn(height, width))

    def forward(self, inputs=None):
        return self.design_domain


class CNNModel(nn.Module):
    """
    Class that implements the CNN model from the structural
    optimization paper
    """

    def __init__(  # noqa
        self,
        args,
        latent_size=128,
        dense_channels=32,
        resizes=(1, 2, 2, 2, 1),
        conv_filters=(128 * 2 // 3, 64 * 2 // 3, 32 * 2 // 3, 16 * 2 // 3, 8 * 2 // 3),
        offset_scale=10.0,
        kernel_size=(12, 12),
        latent_scale=1.0,
        dense_init_scale=1.0,
        random_seed=0,
    ):
        super().__init__()
        set_seed(random_seed)

        # Raise an error if the resizes are not equal to the convolutional
        # filters
        if len(resizes) != len(conv_filters):
            raise ValueError("resizes and filters are not the same!")

        total_resize = int(np.prod(resizes))
        self.h = int(
            torch.div(args["nely"], total_resize, rounding_mode="floor").item()
        )
        self.w = int(
            torch.div(args["nelx"], total_resize, rounding_mode="floor").item()
        )
        self.dense_channels = dense_channels
        self.resizes = resizes
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.latent_size = latent_size
        self.dense_init_scale = dense_init_scale
        self.offset_scale = offset_scale

        # Create the filters
        filters = dense_channels * self.h * self.w

        # Create the first dense layer
        self.dense = nn.Linear(latent_size, filters)

        # # Create the gain for the initializer
        gain = self.dense_init_scale * np.sqrt(max(filters / latent_size, 1.0))
        nn.init.orthogonal_(self.dense.weight, gain=gain)

        # Create the convolutional layers that will be used
        self.conv = nn.ModuleList()

        # Global normalization layers
        self.global_normalization = nn.ModuleList()

        # Trainable bias layer
        self.add_offset = nn.ModuleList()

        # Add the convolutional layers to the module list
        height = self.h
        width = self.w

        dense_channels_tuple = (dense_channels,)
        offset_filters_tuple = conv_filters[:-1]
        offset_filters = dense_channels_tuple + offset_filters_tuple

        for resize, in_channels, out_channels in zip(
            self.resizes, offset_filters, conv_filters
        ):
            convolution_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                padding="same",
            )
            # This was the best initialization
            torch.nn.init.kaiming_normal_(
                convolution_layer.weight, mode="fan_in", nonlinearity="leaky_relu"
            )

            torch.nn.init.zeros_(convolution_layer.bias)
            self.conv.append(convolution_layer)
            self.global_normalization.append(GlobalNormalization())

            # TODO: Fix the offset layer
            height = height * resize
            width = width * resize
            offset_layer = AddOffset(
                scale=self.offset_scale,
                conv_channels=out_channels,
                height=height,
                width=width,
            )
            self.add_offset.append(offset_layer)

        # Set up z here otherwise it is not part of the leaf tensors
        self.z = get_seeded_random_variable(latent_size, random_seed)
        self.z = torch.mean(self.z, axis=0)

        self.z = nn.Parameter(self.z)

    def forward(self, x=None):  # noqa
        # Create the model
        output = self.dense(self.z)
        output = output.reshape((1, self.dense_channels, self.h, self.w))

        layer_loop = zip(self.resizes, self.conv_filters)
        for idx, (resize, filters) in enumerate(layer_loop):
            output = torch.sin(output)
            # After a lot of investigation the outputs of the upsample need
            # to be reconfigured to match the same expectation as tensorflow
            # so we will do that here. Also, interpolate is teh correct
            # function to use here
            output = Fun.interpolate(
                output,
                scale_factor=resize,
                mode="bilinear",
                align_corners=False,
            )

            # Apply the normalization
            output = self.global_normalization[idx](output)

            # Apply the 2D convolution
            output = self.conv[idx](output)

            if self.offset_scale != 0:
                output = self.add_offset[idx](output)

        # # Squeeze the result in the first axis just like in the
        # # tensorflow code
        output = torch.mean(output, axis=1)  # Along the feature dimension
        output = torch.squeeze(output)

        return output


class MultiMaterialCNNModel(nn.Module):
    """
    Class that implements the CNN model from the structural
    optimization paper
    """

    def __init__(  # noqa
        self,
        args: dict,
        latent_size: int = 128,
        dense_channels: int = 32,
        average_pool_size: int = 8,
        resizes: Tuple = (1, 2, 2, 2, 1),
        conv_filters: Tuple = (128, 64, 32, 16),
        offset_scale: float = 10.0,
        kernel_sizes: list[Tuple[int, int]] = [(5, 5), (5, 5), (5, 5), (5, 5), (5, 5)],
        dense_init_scale: float = 1.0,
        random_seed: int = 0,
    ):
        super().__init__()
        set_seed(random_seed)
        self.args = args
        self.multiplier = average_pool_size

        # Update the convolutional filters for the expected
        # number of material channels
        self.num_materials = len(args['e_materials']) + 1
        conv_filters = conv_filters + (self.num_materials * self.multiplier,)

        # Raise an error if the resizes are not equal to the convolutional
        # filteres
        if len(resizes) != len(conv_filters):
            raise ValueError("resizes and filters are not the same!")

        total_resize = int(np.prod(resizes))
        self.h = int(
            torch.div(args["nely"], total_resize, rounding_mode="floor").item()
        )
        self.w = int(
            torch.div(args["nelx"], total_resize, rounding_mode="floor").item()
        )
        self.dense_channels = dense_channels
        self.resizes = resizes
        self.conv_filters = conv_filters
        self.kernel_sizes = kernel_sizes
        self.latent_size = latent_size
        self.dense_init_scale = dense_init_scale
        self.offset_scale = offset_scale

        # Create the filters
        filters = dense_channels * self.h * self.w

        # Create the first dense layer
        self.dense = nn.Linear(latent_size, filters)

        # # Create the gain for the initializer
        gain = self.dense_init_scale * np.sqrt(max(filters / latent_size, 1.0))

        # Still the best initialization for the first layer
        nn.init.orthogonal_(self.dense.weight, gain=gain)
        nn.init.zeros_(self.dense.bias)

        # Create the convoluational layers that will be used
        self.conv = nn.ModuleList()

        # Global normalization layers
        self.global_normalization = nn.ModuleList()

        # Trainable bias layer
        self.add_offset = nn.ModuleList()

        # Add the convolutional layers to the module list
        height = self.h
        width = self.w

        dense_channels_tuple = (dense_channels,)
        offset_filters_tuple = conv_filters[:-1]
        offset_filters = dense_channels_tuple + offset_filters_tuple

        # Performed very well!
        # kernel_sizes = [(5, 5), (5, 5), (7, 7), (9, 9), (9, 9)]
        # kernel_sizes = [(5, 5), (5, 5), (9, 9), (11, 11), (11, 11)]
        # kernel_sizes = [(5, 5), (5, 5), (9, 9), (9, 9)]
        # kernel_sizes = [(7, 7), (7, 7), (11, 11), (11, 11)]
        # kernel_sizes = [(3, 3), (3, 3), (5, 5), (5, 5)]

        for resize, in_channels, out_channels, kernel_size in zip(
            resizes, offset_filters, conv_filters, kernel_sizes
        ):
            convolution_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            )

            # This was the best initialization
            # TODO: Since we are using a sin activation layer I will add
            # the SIREN initialization
            torch.nn.init.kaiming_normal_(
                convolution_layer.weight,
                mode="fan_in",
                nonlinearity="leaky_relu",
            )
            nn.init.zeros_(convolution_layer.bias)  # type: ignore

            self.conv.append(convolution_layer)
            self.global_normalization.append(GlobalNormalization())

            # TODO: Fix the offset layer
            height = height * resize
            width = width * resize
            offset_layer = AddOffset(
                scale=self.offset_scale,
                conv_channels=out_channels,
                height=height,
                width=width,
            )
            self.add_offset.append(offset_layer)

        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.avg_output = nn.AvgPool2d(
            kernel_size=(1, self.multiplier), stride=(1, self.multiplier), padding=0
        )
        self.softmax = nn.Softmax(dim=0)

        # Set up z here otherwise it is not part of the leaf tensors
        self.z = get_seeded_random_variable(latent_size, random_seed)
        self.z = torch.mean(self.z, axis=0)  # type: ignore
        self.z = nn.Parameter(self.z)

    def forward(self, x=None):  # noqa
        # Create the model
        output = self.dense(self.z)
        output = output.reshape((1, self.dense_channels, self.h, self.w))

        layer_loop = zip(self.resizes, self.conv_filters)
        for idx, (resize, _) in enumerate(layer_loop):
            output = torch.sin(output)

            # Upsample interpolation
            output = Fun.interpolate(
                output,
                scale_factor=resize,
                mode="bilinear",
                align_corners=False,
            )

            # Apply the normalization
            output = self.global_normalization[idx](output)

            # Apply the 2D convolution
            output = self.conv[idx](output)

            if self.offset_scale != 0:
                output = self.add_offset[idx](output)

        # The final output will have num_materials + 1 (void)
        # output = self.avg_pool(output)
        output = torch.squeeze(output)

        if self.multiplier != 1:
            output = (
                output.permute(0, 2, 1)
                .reshape(self.num_materials * self.multiplier, -1)
                .t()
            )
            output = self.avg_output(output[None, None, ...]).squeeze()
            output = (
                output.t()
                .reshape(self.num_materials, self.args['nelx'], self.args['nely'])
                .permute(0, 2, 1)
            )

        output = torch.exp(output)
        normalization = torch.norm(output, p=1, dim=0)
        output = output / normalization[None, :]

        # output = self.softmax(output)

        return output


class TopologyOptimizationMLP(nn.Module):
    """
    Class that creates a topology optimization MLP for
    multi-material problem
    """

    def __init__(
        self, num_layers, num_neurons, nelx, nely, num_materials, seed: int = 1234
    ):
        self.input_dim = 2
        # x and y coordn of the point
        self.output_dim = num_materials + 1
        # if material A/B/.../void at the point
        self.nelx = nelx
        self.nely = nely
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = self.input_dim
        set_seed(seed)

        # Iterate the create the layers
        for lyr in range(num_layers):
            l = nn.Linear(current_dim, num_neurons)  # noqa
            nn.init.xavier_uniform_(l.weight)
            nn.init.zeros_(l.bias)
            self.layers.append(l)
            current_dim = num_neurons

        self.layers.append(nn.Linear(current_dim, self.output_dim))
        self.bnLayer = nn.ModuleList()
        for lyr in range(num_layers):
            self.bnLayer.append(nn.BatchNorm1d(num_neurons))

        # Generate the points
        self.x = self._generate_points()

    def _generate_points(self):
        """
        Function to generate the points and make them a part of the neural
        network
        """
        ctr = 0
        xy = np.zeros((self.nelx * self.nely, 2))
        for i in range(self.nelx):
            for j in range(self.nely):
                xy[ctr, 0] = i + 0.5
                xy[ctr, 1] = j + 0.5
                ctr += 1
        xy = torch.tensor(xy, requires_grad=True).float().view(-1, 2)
        xy = nn.Parameter(xy)
        return xy

    def forward(self, fixedIdx=None):  # noqa
        # activations ReLU, ReLU6, ELU, SELU, PReLU, LeakyReLU, Sigmoid,
        # Tanh, LogSigmoid, Softplus, Softsign,
        # TanhShrink, Softmin, Softmax
        m = nn.ReLU6()
        #
        ctr = 0

        # TODO: Why this line? Gather inputs?
        xv = self.x[:, 0]
        yv = self.x[:, 1]

        x = torch.transpose(torch.stack((xv, yv)), 0, 1)
        # Compute each layer
        for layer in self.layers[:-1]:
            x = m(self.bnLayer[ctr](layer(x)))
            ctr += 1
        out = 1e-4 + torch.softmax(self.layers[-1](x), dim=1)

        if fixedIdx is not None:
            out[fixedIdx, 0] = 0.95
            out[fixedIdx, 1:] = 0.01
            # fixed Idx removes region

        return out
