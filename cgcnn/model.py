from typing import Union, Tuple
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from tqdm import trange, tqdm
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_max_pool as gmax
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import global_add_pool, BatchNorm
from torch_scatter import scatter

def gmin(x, batch):
    size = int(batch.max().item() + 1)
    return scatter(x, batch, dim=0, dim_size=size, reduce='min')


class CGConv(MessagePassing):
    def __init__(self, nbr_fea_len, atom_fea_len, orbital,
                 aggr: str = 'add', bias: bool = True, **kwargs):
        super(CGConv, self).__init__(aggr=aggr, flow='target_to_source', **kwargs)


        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        if orbital == True:

            self._mlp = nn.Sequential(
                nn.Linear(50, 35),
                nn.Softplus(),
                nn.Linear(35, 25),
                nn.Softplus() )
            self.lin =  Linear(51, 50)
        else:
            self._mlp = nn.Sequential(
                nn.Linear(128, 96),
                nn.Softplus(),
                nn.Linear(96, 64),
                nn.Softplus() )
            self.lin =  Linear(169, 128)

        self.BatchNorm = BatchNorm(self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.lin_s = Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                self.atom_fea_len)
        self.lin_f = Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                self.atom_fea_len)






    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None):
        """"""
        x:PairTensor = (x, x)
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)

        out, edge_attr = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.softplus(self.BatchNorm(out) + x[1])
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor):
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        if original_ == False:
            return self._mlp(self.lin(z))
        else:
            return self.sigmoid(self.lin_f(z)) * self.softplus(self.lin_s(z))

    def update(self, x, edge_attr):
        return x, edge_attr

class CrystalGraphConvNet(nn.Module):

    def __init__(self, orig_atom_fea_len, nbr_fea_len, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False, orbital = False, original = False, nn_pool = False):

        global original_
        original_ = original
        if orbital == True:
            atom_fea_len = 25
        else:
            atom_fea_len = 64

        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([CGConv(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len, orbital = orbital)
                                    for _ in range(n_conv)])

        self.conv_to_fc_softplus = nn.Softplus()
        self.conv_to_fc = nn.Linear(atom_fea_len*2, h_fea_len)

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])

        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

        self.pool_lin_max = nn.Linear(atom_fea_len, atom_fea_len)
        self.pool_lin_min = nn.Linear(atom_fea_len, atom_fea_len)


    def forward(self, data, orbital, nn_pool):
        atom_fea, bond_index, bond_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if orbital == False:
            atom_fea = self.embedding(atom_fea)

        for i, conv_func in enumerate(self.convs):
            atom_fea = conv_func(x=atom_fea, edge_index=bond_index, edge_attr=bond_attr)

        # MEAN cat MAX POOL
        if nn_pool == True:
            crys_fea = torch.cat([gmax(self.pool_lin_max(atom_fea), batch), gmin(self.pool_lin_min(atom_fea), batch)], dim=1)
        else:
            crys_fea = torch.cat([gap(atom_fea, batch), gmp(atom_fea, batch)], dim=1)
        crys_fea_inter = crys_fea

        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)

        return out, crys_fea_inter
