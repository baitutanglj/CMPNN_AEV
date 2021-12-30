import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from rdkit import Chem                          # 引入化学信息学的包rdkit
from rdkit.Chem import GetAdjacencyMatrix       # 构建分子邻接矩阵
from scipy.sparse import coo_matrix             # 转化成COO格式
import csv
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from typing import List, Optional
from sklearn.metrics import r2_score, mean_squared_error
from argparse import ArgumentParser, Namespace
from scaler import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau,ExponentialLR
from utils import build_optimizer, build_lr_scheduler


import matplotlib.pyplot as plt
def plot_losses(losses, title):
    plt.title(title)
    plt.plot(np.arange(len(losses)), losses)
    plt.show()

# data_path = '/mnt/home/linjie/projects/CMPNN-master-copy/pdbbind2016/pdbbind_v2016_refined/refined_2016_pathdata.csv'
# '/mnt/home/linjie/projects/CMPNN-master-copy/pdbbind2016/pdbbind_v2016_refined/small_pdb.dat'
def add_args(parser: ArgumentParser):
    """
    Adds dataload and training arguments to an ArgumentParser.
    :param parser: An ArgumentParser.
    """
    # Create two argument groups
    data_group = parser.add_argument_group(title='Add dataload argments to an ArgumentParser')
    train_group = parser.add_argument_group(title='Add training argments to an ArgumentParser')

    # Add arguments to data_group
    data_group.add_argument('--data_path', type=str, help='Path to data CSV file',
                            default='/mnt/home/linjie/projects/CMPNN-master-copy/pdbbind2016/pdbbind_v2016_refined/refined_2016_pathdata.csv')
    data_group.add_argument('--test_path', type=str, help='Path to data CSV file',
                            default='/mnt/home/linjie/projects/CMPNN-master-copy/pdbbind2016/CASF-2016/CoreSetPath.csv')
    data_group.add_argument('--pdbbind_path', type=str, help=' path to pdbbind data',
                            default = '/mnt/home/linjie/projects/CMPNN-master-copy/pdbbind2016/pdbbind_v2016_refined/refined-set')
    data_group.add_argument('--features_path', type=str, help='features_path',
                            default='/mnt/home/linjie/projects/CMPNN-master-copy/aevs_descriptor/result3.5/train_aevs.pkl')
    data_group.add_argument('--test_features_path', type=str, help='test_features_path',
                            default='/mnt/home/linjie/projects/CMPNN-master-copy/aevs_descriptor/result3.5/test_aevs.pkl')
    data_group.add_argument('--smiles_columns', type=str, default='smiles',
                        help='Name of the columns containing SMILES strings. Default:smiles')
    data_group.add_argument('--target_columns', type=List, default=['affinity'],
                        help='Name of the columns containing target values. Default:["affinity"]')

    # Add arguments to train_group
    train_group.add_argument('--epochs', type=int, default=300, help='Number of epochs to run')
    train_group.add_argument('--batch_size', type=int ,default=285, help='Batch size')
    train_group.add_argument('--hidden_size', type=int, default=512, help='Dimensionality of hidden layers in Graph model')
    train_group.add_argument('--dnn_layers', type=int, nargs='+', default=[512,300,1], help="DNN model layers")
    train_group.add_argument('--dropout', type=float, default=0.25, help='Dropout probability')
    train_group.add_argument('--gpu', type=int,choices=list(range(torch.cuda.device_count())), default=0, help='Which GPU to use')
    train_group.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    train_group.add_argument('--save_path', type=str, default='/mnt/home/linjie/projects/CMPNN-master-copy/pyG/result',
                             help='save the best model states path')


# Load data
def load_dataset(datapath: str, features_path:str,  args: Namespace,
                 shuffle=True, drop_last=True) -> DataLoader:
    skip_smiles = set()
    skip_none_targets = True
    features_df = pd.read_pickle(features_path)
    features = features_df['features']
    input_dim = features[0].shape[1]
    data_list = []
    with open(datapath) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(tqdm(reader)):
            smiles = row[args.smiles_columns]
            if smiles in skip_smiles:
                continue
            targets = [float(row[column]) for column in args.target_columns]

            # Check whether all targets are None and skip if so
            if skip_none_targets and all(x is None for x in targets):
                continue
            mol = Chem.SDMolSupplier(args.pdbbind_path + '/' + smiles)[0]
            adj = GetAdjacencyMatrix(mol)  # 创建邻接矩阵
            coo_adj = coo_matrix(adj)
            edge_index = [coo_adj.row, coo_adj.col]

            x = torch.from_numpy(features[smiles.split('/')[0]])
            # y = torch.tensor(targets)
            y = targets
            scope = torch.tensor([x.size(0)])
            edge_index = torch.tensor(edge_index, dtype=torch.long)  # 这里的dtype需要注意一下
            data = Data(x=x, y=y, edge_index=edge_index, scope=scope)
            data_list.append(data)

    # 生成data loader
    loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader, input_dim

class GCN_Net(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_size: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        #atoms to mol
        x = torch.split(x, data.scope.tolist(), dim=0)
        mol_vecs = [mol_vec.mean(0) for mol_vec in x]
        mol_vecs = torch.stack(mol_vecs, dim=0)

        return mol_vecs



class GAT_Net(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_size: int, 
                 heads: int=1):
        super(GAT_Net, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_size, heads=heads)
        self.bn1 = nn.BatchNorm1d(hidden_size*heads)
        self.gat2 = GATConv(hidden_size*heads, hidden_size, heads=heads)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)
        
        # atoms to mol
        x = torch.split(x, data.scope.tolist(), dim=0)
        mol_vecs = [mol_vec.mean(0) for mol_vec in x]
        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs


class GraphSAGE_Net(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_size: int):
        super(GraphSAGE_Net, self).__init__()
        self.sage1 = SAGEConv(input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.sage2 = SAGEConv(hidden_size, hidden_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.sage1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)

        # atoms to mol
        x = torch.split(x, data.scope.tolist(), dim=0)
        mol_vecs = [mol_vec.mean(0) for mol_vec in x]
        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs

class DNN_Net(nn.Module):
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
        dropout: Optional[float] = None,
    ):

        super().__init__()

        if layers_sizes is None:
            # Default values from TorchANI turorial
            # self.layers_sizes: List[int] = [160, 128, 96, 1]
            self.layers_sizes: List[int] = [256, 128, 64, 1]
        else:
            self.layers_sizes = layers_sizes.copy()

        # Prepend input size to other layer sizes
        self.layers_sizes.insert(0, first_linear_dim)

        self.layers = nn.ModuleList()

        for in_size, out_size in zip(self.layers_sizes[:-2], self.layers_sizes[1:-1]):
            self.layers.append(nn.Linear(in_size, out_size))
            self.layers.append(nn.BatchNorm1d(out_size))
            self.layers.append(nn.ReLU())

            if dropout is not None:
                self.layers.append(nn.Dropout(dropout))

        # Last linear layer
        self.layers.append(nn.Linear(self.layers_sizes[-2], self.layers_sizes[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x



class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a Graph network following by DNN network."""
    def __init__(self, input_dim: int,
                 args: Namespace=None):
        super(MoleculeModel, self).__init__()
        self.args = args
    def create_Graph_model(self, args:Namespace):
        """
        Creates the graph_model.
        :param args: Arguments.
        """
        # self.graph_model = GCN_Net(args.input_dim, args.hidden_size)
        # self.graph_model = GAT_Net(args.input_dim, args.hidden_size, heads=8)
        self.graph_model = GraphSAGE_Net(args.input_dim, args.hidden_size)


    def create_DNN_model(self, args:Namespace):
        """
        Creates the feed-forward network for the model.
        :param args: Arguments.
        """
        self.DNN_model = DNN_Net(first_linear_dim=args.hidden_size,
                                 layers_sizes=args.dnn_layers,
                                 dropout=args.dropout)

    def forward(self, dataload):
        """Runs the MoleculeModel on input."""
        graph_output = self.graph_model(dataload)
        output = self.DNN_model(graph_output)
        return output

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


def initialize_weights(model: nn.Module):
    """
    Initializes the weights of a model in place.

    :param model: An nn.Module.
    """
    # for param in model.parameters():
    #     if param.dim() == 1:
    #         nn.init.constant_(param, 0)
    #     else:
    #         nn.init.xavier_normal_(param)
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_normal_(param)


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a graph neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the graph neural network along with final linear layers with parameters initialized.
    """
    model = MoleculeModel(args)
    model.create_Graph_model(args)
    model.create_DNN_model(args)
    initialize_weights(model)
    return model

def train(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    epoch_loss = 0

    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = batch.to(device)
        # Run model forward
        model.zero_grad()
        preds = model(batch)
        loss = criterion(preds, batch.y.unsqueeze(1))

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    batch_loss = epoch_loss / len(dataloader)
    # scheduler为学习率调整策略,针对loss进行学习率改变。记得加上评价指标loss。这条语句可放在epoch的循环位置，要放在batch循环位置也可以，只是正对patience对象不同。
    # scheduler.step(batch_loss)
    return batch_loss

def val(model, dataloader, criterion, scaler):
    model.eval()
    pred_list = []
    targets_list = []
    eval_loss = 0

    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = batch.to(device)
        # Run model
        preds = model(batch)
        loss = criterion(preds, batch.y.unsqueeze(1))
        eval_loss += loss.item()

        pred_list.extend(preds.data.cpu().tolist())
        targets_list.extend(batch.y.unsqueeze(1).data.cpu().tolist())
    r2 = r2_score(targets_list, pred_list)
    mse = mean_squared_error(targets_list, pred_list)
    batch_eval_loss = eval_loss / len(dataloader)
    pred_list = scaler.inverse_transform(pred_list)
    return batch_eval_loss, r2, pred_list

def run_training(model, trainloader, testloader,
                 criterion, optimizer, args,
                 scaler, scheduler):
    train_losses = []
    eval_losses = []
    all_val_r2 = []
    best_r2 = float('-inf')
    best_epoch = 0
    for epoch in range(args.epochs):
        train_loss = train(model, trainloader, criterion, optimizer, args.gpu)
        scheduler.step(train_loss)
        eval_loss, val_r2, pred_list = val(model, testloader, criterion, scaler)
        print(f'epoch: {epoch}, loss:{train_loss}')
        print(f'epoch: {epoch}, eval_loss:{eval_loss}, val_r2:{val_r2}')
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        all_val_r2.append(val_r2)
        if val_r2 > best_r2:
            best_r2, best_epoch = val_r2, epoch
            best_model = model
    print(f'best r2 :{best_r2}  in  epoch{best_epoch}')

    return train_losses, eval_losses, all_val_r2, best_r2, best_model


def parser_args() -> Namespace:
    """
    Parses arguments for dataload and training.
    :return: A Namespace containing the parsed args.
    """
    parser = ArgumentParser(description=' training Graph model argument')
    add_args(parser)
    args = parser.parse_args()
    return args

def scaler_targets(dataloader, scaler=None):
    targets = [dataloader.dataset[i].y for i in range(len(dataloader.dataset))]
    if scaler is None:
        scaler = StandardScaler().fit(targets)
    scaled_targets_list = scaler.transform(targets).tolist()
    for i in range(len(dataloader.dataset)):
        dataloader.dataset[i].y = torch.tensor(scaled_targets_list[i])
    return dataloader, scaler




if __name__ == '__main__':
    args = parser_args()
    device = args.gpu
    trainloader, args.input_dim = load_dataset(args.data_path, args.features_path, args)
    testloader, _ = load_dataset(args.test_path, args.test_features_path, args,
                                 shuffle=False, drop_last=False)
    trainloader, scaler = scaler_targets(trainloader)
    testloader, _ = scaler_targets(testloader, scaler)

    model = build_model(args).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    # scheduler = ExponentialLR(optimizer, gamma=0.9, verbose=True)
    # scheduler = build_lr_scheduler(optimizer=optimizer,
    #                                warmup_epochs=2.0,
    #                                total_epochs=[args.epochs] * 1,
    #                                steps_per_epoch=len(trainloader.dataset) // args.batch_size,
    #                                init_lr=1e-3,
    #                                max_lr=1e-3,
    #                                final_lr=1e-4)
    criterion = nn.MSELoss()
    train_losses, eval_losses, all_val_r2, best_r2, best_model = run_training(model,
                                                                             trainloader,
                                                                             testloader,
                                                                             criterion,
                                                                             optimizer,
                                                                             args,
                                                                             scaler,
                                                                             scheduler)
    torch.save(best_model.state_dict(), args.save_path+'/model.pt')
    #predict
    model = build_model(args).to(device)
    model.load_state_dict(torch.load(args.save_path+'/model.pt'))
    mse, r2, pred_list = val(model, testloader, criterion, scaler)
    if os.path.exists(args.save_path+'/predict.csv'):
        os.remove(args.save_path+'/predict.csv')
    with open(args.save_path+'/predict.csv', 'w+') as f:
        for i in pred_list:
            f.write(str(i.item())+'\n')
    plot_losses(train_losses,'GraphSAGE train_losses')
    plot_losses(eval_losses, 'GraphSAGE eval_losses')
    plot_losses(all_val_r2, 'GraphSAGE all_val_r2')




