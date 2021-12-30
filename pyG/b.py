# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from rdkit import Chem                          # 引入化学信息学的包rdkit
from rdkit.Chem import GetAdjacencyMatrix       # 构建分子邻接矩阵
from scipy.sparse import coo_matrix             # 转化成COO格式
import torch
from torch_geometric.data import Data           # 引入pyg的Data
import numpy as np


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        print(x,type(x))
        raise Exception("input {0} not in allowable set{1}:".format(
                x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def GetVertMat(atoms):
    V = []
    for a in atoms:
        v = []
        v.extend(one_of_k_encoding(a.GetSymbol(), element_symbol_list))
        v.append(a.GetMass())
        V.append(v)
    return np.array(V)


element_symbol_list = ['Cl', 'N', 'S', 'F', 'C', 'O', 'H']

SMILES = ['Cc1ccccc1',
          'CCO',
          'CC(=O)Nc1ccc(O)cc1',
          'c1cccc(c1OC(=O)C)C(O)=O',
          'CC(=O)C',
          'C1=CC=C(C=C1)[N+](=O)[O-]']

data_list = []
for smiles in SMILES:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    A = GetAdjacencyMatrix(mol)    # 创建邻接矩阵
    coo_A = coo_matrix(A)
    edge_index = [coo_A.row, coo_A.col]
    atoms = mol.GetAtoms()
    V = GetVertMat(atoms)
    # 输入都转换成py torch的tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long)   # 这里的dtype需要注意一下
    x = torch.tensor(V,dtype=torch.float32)
    print(x.shape)
    import random
    y = random.random()    # 这里y用随机数代替，工作中请使用具体的属性
    y = torch.tensor([y])
    # smi = torch.tensor([smiles])
    # Data的属性通过初始化参数自定义，这里我们可以把smiles加入到属性中
    data = Data(x=x, y=y, edge_index=edge_index)
    data_list.append(data)   # 将实例化的Data加入到list中

# 生成data loader
loader = DataLoader(data_list, batch_size=3, shuffle=True)

# 生成batch
for batch in loader:
    print(len(batch))
    print(batch.x)
    print(batch.y)
    print(batch.edge_index)

##################################
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNConv(in_channels=8, out_channels=1).to(device)
pred_list = []
for batch in loader:
    data = batch.to(device)
    pred = model(data.x, data.edge_index)
    pred_list.extend(pred.tolist())