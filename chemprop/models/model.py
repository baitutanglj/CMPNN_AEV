from argparse import Namespace

import torch.nn as nn

from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights
from collections import OrderedDict
from typing import List, Optional
import torch
import numpy as np


class AtomicNN(nn.Module):
    """
    Atomic Neural Network (ANN)

    Parameters
    ----------
    first_linear_dim: int
        Input size (input length)
    layers_sizes: List[int]
        List with the size of fully connected layers, excluding firs
    dropp: Optional[float]
        Dropout probability
    """

    def __init__(
        self,
        first_linear_dim: int,
        layers_sizes: Optional[List[int]] = None,
        dropp: Optional[float] = None,
    ):

        super().__init__()

        if layers_sizes is None:
            # Default values from TorchANI turorial
            self.layers_sizes: List[int] = [160, 128, 96, 1]
        else:
            self.layers_sizes = layers_sizes.copy()

        # Prepend input size to other layer sizes
        self.layers_sizes.insert(0, first_linear_dim)

        self.layers = nn.ModuleList()

        for in_size, out_size in zip(self.layers_sizes[:-2], self.layers_sizes[1:-1]):
            self.layers.append(nn.Linear(in_size, out_size))
            self.layers.append(nn.ReLU())

            if dropp is not None:
                self.layers.append(nn.Dropout(dropp))

        # Last linear layer
        self.layers.append(nn.Linear(self.layers_sizes[-2], self.layers_sizes[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

class FFNModel(nn.ModuleDict):

    def __init__(
            self,
            n_species:int,
            first_linear_dim:int,
            layers_sizes: Optional[List[int]] = None,
            dropp: Optional[float] = None,
    ):
        assert n_species > 0

        modules = n_species * [
            AtomicNN(first_linear_dim, layers_sizes=layers_sizes, dropp=dropp)
        ]

        super().__init__(self.ensureOrderedDict(modules))

        # Store values
        self.first_linear_dim = first_linear_dim
        self.n_species = n_species
        self.layers_sizes = modules[0].layers_sizes
        self.dropp = dropp

    def forward(self, species, input):
        """
        Notes
        -----
        Copyright 2018-2020 Xiang Gao and other ANI developers.
        """
        species_ = species.flatten().cuda()
        input = input.flatten(0, 1)

        output = input.new_zeros(species_.shape)

        for i, (_, m) in enumerate(self.items()):
            mask = species_ == i
            if sum(mask)>0:
                midx = mask.nonzero().flatten()
                if midx.shape[0] > 0:
                    input_ = input.index_select(0, midx)
                    output.masked_scatter_(mask, m(input_).flatten())
        output = output.view_as(species)
        return torch.sum(output, dim=1, keepdim=True)

    @staticmethod
    def ensureOrderedDict(modules):
        """
        Notes
        -----
        Copyright 2018-2020 Xiang Gao and other ANI developers.
        """
        if isinstance(modules, OrderedDict):
            return modules
        od = OrderedDict()
        for i, m in enumerate(modules):
            od[str(i)] = m
        return od


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool, n_species: int=None):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)
        ##############my addition###############
        self.n_species = n_species
        ########################################

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * 1
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        # if args.ffn_num_layers == 1:
        #     ffn = [
        #         dropout,
        #         nn.Linear(first_linear_dim, args.output_size)
        #     ]
        # else:
        #     ffn = [
        #         dropout,
        #         nn.Linear(first_linear_dim, args.ffn_hidden_size)
        #     ]
        #     for _ in range(args.ffn_num_layers - 2):
        #         ffn.extend([
        #             activation,
        #             dropout,
        #             nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
        #         ])
        #     ffn.extend([
        #         activation,
        #         dropout,
        #         nn.Linear(args.ffn_hidden_size, args.output_size),
        #     ])

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            layers_input_size = args.ffn_hidden_size
            layers_output_size = args.ffn_hidden_size
            for i in range(args.ffn_num_layers - 2):
                layers_output_size = int(layers_input_size * 0.5)
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(layers_input_size, layers_output_size),
                ])
                layers_input_size = layers_output_size
                # ffn.extend([
                #     activation,
                #     dropout,
                #     nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                # ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(layers_output_size, args.output_size),
            ])


        # Create FFN model
        ##############my addition##################
        if args.n_species:
            self.ffn = FFNModel(args.n_species, first_linear_dim, args.ffn_layers, args.dropout)
        else:
            self.ffn = nn.Sequential(*ffn)
        ###########################################

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """

        #############my addition##############
        encoder_output = self.encoder(*input[:-1])
        if input[-1][0]:
            output = self.ffn(input[-1], encoder_output)
        else:
            output = self.ffn(encoder_output)
        ######################################

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output
    ##############my addition##############
    @staticmethod
    def ensureOrderedDict(modules):
        """
        Notes
        -----
        Copyright 2018-2020 Xiang Gao and other ANI developers.
        """
        if isinstance(modules, OrderedDict):
            return modules
        od = OrderedDict()
        for i, m in enumerate(modules):
            od[str(i)] = m
        return od
    #########################################
def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification',
                          multiclass=args.dataset_type == 'multiclass',
                          n_species=args.n_species)
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)



    return model
