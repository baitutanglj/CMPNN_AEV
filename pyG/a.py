import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
#>>>Data(x=[3, 1], edge_index=[2, 4]) #3个node，每个nodefeatures_dim=1，num_edges=4

from torch_geometric.datasets import TUDataset
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
len(dataset)
dataset.num_classes
dataset.num_node_features
data = dataset[0]
data.is_undirected()

#####GCN#####
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.datasets import Planetoid

class GCN_Net(torch.nn.Module):
    def __init__(self,features, hidden, classes):
        super().__init__()
        self.conv1 = GCNConv(features, hidden)
        self.conv2 = GCNConv(hidden, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='/home/linjie/Downloads/pyG_data/Cora', name='Cora')
model = GCN_Net(dataset.num_node_features, 16, dataset.num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    print(f'epoch:{epoch}, loss:{loss}')
    loss.backward()
    optimizer.step()

model.eval()
_,pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
accuracy = int(correct) / int(data.test_mask.sum())
print(accuracy)


#####GraphSAGE######
class GraphSAGE_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super(GraphSAGE_Net, self).__init__()
        self.sage1 = SAGEConv(features, hidden)
        self.sage2 = SAGEConv(hidden, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)

        return F.log_softmax(x, dim=1)

model = GraphSAGE_Net(dataset.num_node_features, 16, dataset.num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    print(f'epoch:{epoch}, loss:{loss}')
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
accuracy = int(correct) / int(data.test_mask.sum())
print(accuracy)


#####GAT######
class GAT_Net(torch.nn.Module):
    def __init__(self,features, hidden, classes, heads=1):
        super(GAT_Net, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=heads)
        self.gat2 = GCNConv(hidden*heads, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1)


model = GAT_Net(dataset.num_node_features, 16, dataset.num_classes, heads=4).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    print(f'epoch:{epoch}, loss:{loss}')
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
accuracy = int(correct) / int(data.test_mask.sum())
print('GAT', accuracy)


def gcn(self, input, adj):
    # input =[batch,num_node, atom_dim]
    # adj = [batch,num_node, num_node]
    support = torch.matmul(input, self.weight)
    # support =[batch,num_node,atom_dim]
    output = torch.bmm(adj, support)
    # output = [batch,num_node,atom_dim]
    return output


def forward(self, compound, adj):
    compound = self.gcn(compound, adj)


from rdkit import Chem                          # 引入化学信息学的包rdkit
from rdkit.Chem import GetAdjacencyMatrix       # 构建分子邻接矩阵
from scipy.sparse import coo_matrix             # 转化成COO格式
import torch
from torch_geometric.data import Data           # 引入pyg的Data
import numpy as np
from scipy.sparse import coo_matrix
mol = Chem.MolFromSmiles('CCO')
mol = Chem.AddHs(mol)
A = GetAdjacencyMatrix(mol)
coo_A = coo_matrix(A)
edge_index = [coo_A.row, coo_A.col]



from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)