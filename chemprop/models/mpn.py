from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x).transpose(0, 1)


class MPNEncoder(nn.Module):
    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        #Batch Normalization
        # self.batchNorma_layer = nn.BatchNorm1d(num_features)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Input
        input_dim = self.atom_fdim
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        input_dim = self.bond_fdim
        self.W_i_bond = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        
        
        w_h_input_size_atom = self.hidden_size + self.bond_fdim
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)
        
        w_h_input_size_bond = self.hidden_size
        
        
        for depth in range(self.depth-1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)
            # output_size_bond = int(w_h_input_size_bond*0.5)
            # self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, output_size_bond, bias=self.bias)
            # w_h_input_size_bond = int(output_size_bond*0.5)

        # self.lr = nn.Linear(self.hidden_size*2+output_size_bond, self.hidden_size, bias=self.bias)
        self.lr = nn.Linear(self.hidden_size*3, self.hidden_size, bias=self.bias)

        self.gru = BatchGRU(self.hidden_size)

        self.W_o = nn.Linear(
            (self.hidden_size) * 2,
            self.hidden_size)
        ##############transformer#################
        # self.positional_encoding = PositionalEncoding(d_model=self.hidden_size)
        # self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size,
        #                                                             dim_feedforward=1024,
        #                                                             nhead=8,
        #                                                             batch_first=True,
        #                                                             device=self.args.gpu,
        #                                                             )
        ##########################################

    def forward(self,mol_graph: BatchMolGraph, split_model:bool=False) -> torch.FloatTensor:

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds = mol_graph.get_components()
        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = (
                    f_atoms.cuda(self.args.gpu)  , f_bonds.cuda(self.args.gpu)  ,
                    a2b.cuda(self.args.gpu)  , b2a.cuda(self.args.gpu)  , b2revb.cuda(self.args.gpu)  )
            
        # Input
        input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()
        
        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)
        # Message passing
        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            message_atom = message_atom + agg_message
            
            # directed graph
            rev_message = message_bond[b2revb]  # num_bonds x hidden
            message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden
            
            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))
        
        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru(agg_message, a_scope)
        
        atom_hiddens = self.act_func(self.W_o(agg_message))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        
        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            # norm = torch.norm(cur_hiddens, dim=1)
            # norm = F.softmax(norm)
            # norm = norm.unsqueeze(1)
            # cur_hiddens = torch.sum(cur_hiddens*norm,dim=0)
            if split_model:
                mol_vecs.append(cur_hiddens)
            elif self.args.use_transformer:
                mol_vecs.append(cur_hiddens)
            else:
                mol_vecs.append(cur_hiddens.mean(0))

        if split_model:
            mol_vecs = nn.utils.rnn.pad_sequence(
                mol_vecs,batch_first=True,padding_value=-1
            )
        elif self.args.use_transformer:
            return mol_vecs  # B x H
        else:
            mol_vecs = torch.stack(mol_vecs, dim=0)
        
        return mol_vecs  # B x H


class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru  = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                           bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 
                                1.0 / math.sqrt(self.hidden_size))


    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            
            cur_message = torch.nn.ZeroPad2d((0,0,0,MAX_atom_len-cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))
            
        message_lst = torch.cat(message_lst, 0)
        hidden_lst  = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2,1,1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)
        
        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2*self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)
        
        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1), 
                             cur_message_unpadding], 0)
        return message


class MPN(nn.Module):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False,
                 another_model: bool = False):
        super(MPN, self).__init__()
        self.args = args
        ###############my addition#######
        self.use_input_features = args.use_input_features
        self.features_only = args.features_only
        self.device = args.device
        self.another_model = another_model
        self.split_model = args.split_model
        ##################################
        self.atom_fdim = atom_fdim or get_atom_fdim(args.overwrite_default_atom_features)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + \
                            (not args.atom_messages) * self.atom_fdim # * 2
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self, batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None,
                ) -> torch.FloatTensor:

        ###################my addition####################
        if self.split_model and self.use_input_features:
            raise ValueError('Currently cannot provide both use split_model and use_input_features in FFN model.')
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float().to(self.device)
            if self.features_only:
                return features_batch

        if not self.graph_input:  # if features only, batch won't even be used
            # batch = mol2graph(batch, self.args)
            batch = mol2graph(batch, self.args,
                              atom_descriptors_batch,
                              bond_features_batch,
                              self.another_model)
        ###################################################

        # split_model==True-->output:(mol_num,max_atoms_num,feature_size)
        # split_model==False-->output:(mol_num,feature_size)
        output = self.encoder.forward(batch, self.split_model)

        ##############my addition##############
        if self.split_model:
            return output

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)

            output = torch.cat([output, features_batch], dim=1)
        ########################################

        return output
