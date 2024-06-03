import torch
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, GCNConv, GCN2Conv
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops
import numpy as np
from torch_geometric.utils import dropout_adj
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from torch import Tensor
from utils import edge_index_to_adj
from einops import rearrange, repeat
class LogReg(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()
      
    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
# class GCN(MessagePassing):
#     def __init__(self, in_channels: int, hidden_channels: int):
#         super(GCN, self).__init__()
#         self.lin1 = nn.Linear(in_channels, hidden_channels)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.lin1.reset_parameters()
#
#     def forward(self, x, edge_index, edge_weight=None, K=1):
#         edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)
#
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.lin1(x)
#         for k in range(K):
#             x = self.propagate(edge_index, x=x, norm=norm)
#         return x
#
#     def message(self, x_j, norm):
#         return norm.view(-1, 1) * x_j
class Interaction_Attention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCN(MessagePassing):
    def __init__(self, in_channels: int, hidden_channels: int):
        super(GCN, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, K=1):
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.lin1(x)
        for k in range(K):
            x = self.propagate(edge_index, x=x, norm=norm)
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
class TransformerModel(nn.Module):

    def __init__(self,  in_dim: int, proj_dim:int, n_head: int, ff_dim: int, out_dim:int, nlayers: int, dropout: float = 0.5,hops=3,activation='relu'):
        super().__init__()
        self.num_hop = hops
        self.fc = Linear(in_dim,proj_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1,hops,proj_dim))
        self.dropout = nn.Dropout(dropout)
        self.build_activation(activation)
        # encoder_layers = TransformerEncoderLayer(proj_dim, n_head, ff_dim, dropout=0.1, batch_first=True)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.build_feature_inter_layer(proj_dim, nlayers)

        # self.norm_layer = nn.LayerNorm(proj_dim)
        self.build_norm_layer(proj_dim, nlayers * 2 + 2)
        self.projector = nn.Linear(proj_dim, out_dim)


        # self.init_weights()
    def build_activation(self, activation):
        if activation == 'tanh':
            self.activate = F.tanh
        elif activation == 'sigmoid':
            self.activate = F.sigmoid
        elif activation == 'gelu':
            self.activate = F.gelu
        else:
            self.activate = F.relu
    def init_weights(self) -> None:
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.projector.bias.data.zero_()
        self.projector.weight.data.uniform_(-initrange, initrange)

    def build_feature_inter_layer(self, hidden_channels, inter_layer):
        self.interaction_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        for i in range(inter_layer):
            self.interaction_layers.append(
                Interaction_Attention(hidden_channels, heads=4, dropout=0.1))

    def build_norm_layer(self, hidden_channels, layers):
        self.norm_layers = nn.ModuleList()
        for i in range(layers):
            self.norm_layers.append(nn.LayerNorm(hidden_channels))


    def norm(self, h, layer_index):
        h = self.norm_layers[layer_index](h)
        return h

    def build_hops(self,x,edge_index):
        h0=[]
        adj = edge_index_to_adj(edge_index, num_nodes=x.size(0))
        for i in range(self.num_hop):
            h0.append(x)
            x = adj@ x
        self.h0 =  torch.stack(h0, dim=1)
        return self.h0
    # def norm(self, h):
    #     h = self.norm_layer(h)
    #     return h

    def embedding(self, h):
        h = self.dropout(h)
        h = self.fc(h)
        h = h + self.pos_encoder
        h = self.norm(h, 0)
        return h

    def interaction(self, h):
        inter_layers = len(self.interaction_layers)
        for i in range(inter_layers):
            h_prev = h
            h = self.dropout(h)
            h = self.interaction_layers[i](h)
            h = self.activate(h)
            h = h + h_prev
            h = self.norm(h, i + 1)
        return h

    def fusion(self,h):
        h = self.dropout(h)
        h = h.mean(dim=1)
        return h

    def forward(self, x,edge_index, stop_hops=False) :
        if stop_hops:
            h0 = []
            adj = edge_index_to_adj(edge_index, num_nodes=x.size(0))
            for i in range(2):
                h0.append(x)
                x = adj @ x
            h = torch.stack(h0, dim=1)
            h = self.dropout(h)
            h = self.fc(h)
            h = h + self.pos_encoder[:,:2,:]
            h = self.norm(h, 0)

        else:
            h = self.build_hops(x,edge_index)
            h = self.embedding(h)

        h = self.interaction(h)
        output = self.fusion(h)
        return output

    def emb(self,x,edge_index,stop_hops=False):
        if stop_hops:
            h0 = []
            adj = edge_index_to_adj(edge_index, num_nodes=x.size(0))
            for i in range(2):
                h0.append(x)
                x = adj @ x
            h = torch.stack(h0, dim=1)
            h = self.dropout(h)
            h = self.fc(h)
            h = h + self.pos_encoder[:,:2,:]
            h = self.norm(h, 0)

        else:
            h = self.build_hops(x,edge_index)
            h = self.embedding(h)
        # h = self.build_hops(x,edge_index)
        # h = self.embedding(h)

        output = self.interaction(h)
        output = self.fusion(output)
        return output


class Model(torch.nn.Module):
    def __init__(self, num_features, num_hidden: int, tau1: float, tau2: float, l1: float, l2: float, n_head=4, ff_dim=2048,out_dim=256,nlayers=2,dropout=0.5, hops=3,global_structure_hops=10):
        super(Model, self).__init__()
        self.tau1: float = tau1
        self.tau2: float = tau2
        self.gcn_s1 = GCN(num_features, num_hidden)
        self.gcn_s2 = TransformerModel(in_dim = num_features, proj_dim = num_hidden, n_head = n_head, ff_dim=ff_dim, out_dim=out_dim, nlayers=nlayers, dropout=dropout,hops=global_structure_hops)
        self.gcn_f1 = GCN(num_features, num_hidden)
        self.gcn_f2 = GCN(num_features, num_hidden)

        self.mlp = Linear(num_hidden*2,1)
        self.l1 = l1
        self.l2 = l2
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.gcn_s1.reset_parameters()
        # self.gcn_s2.reset_parameters()
        # self.gcn_f1.reset_parameters()
        self.gcn_f2.reset_parameters()


    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor, knn_graph: torch.Tensor) -> torch.Tensor:
        
        h0 = self.gcn_s1(x, edge_index,K=2)
        h1 = self.gcn_s2(x, edge_index)##cora & citeseer 10, pubmed 30
  
        z0 = self.gcn_f2(x, edge_index, K=2)
        z1 = self.gcn_f2(x, knn_graph, K=1)

        y = self.mlp(torch.cat([h1,z1],dim=-1))
        return h0, h0+h1, z0, z0+z1,y

    def embed(self, x: torch.Tensor,
                edge_index: torch.Tensor, knn_graph) -> torch.Tensor:

        h0 = self.gcn_s1(x, edge_index, K=2)
        h1 = self.gcn_s1(x, edge_index, K=2)+self.gcn_s2.emb(x, edge_index)  ##cora & citeseer 10, pubmed 30

        z0 = self.gcn_f2(x, edge_index, K=2)
        z1 = self.gcn_f2(x, edge_index, K=2) + self.gcn_f2(x, knn_graph, K=1)

        return (h0 + h1 + z1).detach()
        #return (z1 + z0).detach()

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    
    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau1)
        between_sim = f(self.sim(z1, z2))
        refl_sim1 = f(self.sim(z1, z1))
        refl_sim2 = f(self.sim(z2, z2))
  
        return (-torch.log(
            between_sim.diag()
            / (refl_sim1.sum(1) + between_sim.sum(1) - refl_sim1.diag() + refl_sim2.sum(1) - refl_sim2.diag()))).mean()


    def loss(self, h0, h1, z0, z1):
        l1 = self.semi_loss(h0, h1) 
        l2 = self.semi_loss(z0, z1)
        co_l = self.semi_loss(h0, z0)
        
        return l1 + self.l1 * l2 + self.l2 * co_l 
        #return self.l1 * l2

    def disloss(self, edge_index, knn_graph,y):
        A = edge_index_to_adj(edge_index,y.size(0)) >0
        A1 = edge_index_to_adj(knn_graph,y.size(0)) >0

        label = torch.sum(A&A1,dim=1)/torch.sum(A|A1,dim=1)

        return nn.MSELoss()(label,y)

    # def strucloss(self):