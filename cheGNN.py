import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np
from torch import nn
from torch.nn import init

import math
import dgl
import sympy
import scipy

from dgl.nn.pytorch import GraphConv, EdgeWeightNorm, ChebConv, GATConv, HeteroGraphConv


class ChebyshevConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k,  
                 activation=F.leaky_relu,
                 bias=False):
        super(ChebyshevConv, self).__init__()
        self._k = k
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        
        self.theta = nn.Parameter(torch.Tensor(k + 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.theta, mean=0, std=0.1)
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def chebyshev_laplacian(feat, D_invsqrt, graph, k):
            """计算切比雪夫多项式 T_k(L~)x """
            L = lambda x: unnLaplacian(x, D_invsqrt, graph)
            
            # T_0(L~)x = x
            if k == 0:
                return feat
            
            # T_1(L~)x = L~x
            if k == 1:
                return L(feat)
            
            # T_k(L~)x = 2L~ T_{k-1}(L~)x - T_{k-2}(L~)x
            t_k_minus_1 = L(feat)
            t_k_minus_2 = feat
            for _ in range(2, k + 1):
                t_k = 2 * L(t_k_minus_1) - t_k_minus_2
                t_k_minus_2, t_k_minus_1 = t_k_minus_1, t_k
            return t_k

        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation: L~x = (I - D^-1/2 A D^-1/2)x """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            
            h = self.theta[0] * feat
            for k in range(1, self._k + 1):
                h += self.theta[k] * chebyshev_laplacian(feat, D_invsqrt, graph, k)
            
            h = self.linear(h)
            if self.activation is not None:
                h = self.activation(h)
        return h

class DomainAdjustedChebyshevConv(nn.Module):
    def __init__(self, in_feats, out_feats, k, activation=F.leaky_relu, bias=False):
        super(DomainAdjustedChebyshevConv, self).__init__()
        self._k = k
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        
        self.theta = nn.Parameter(torch.Tensor(k + 1))
        self.reset_parameters()
   
    def reset_parameters(self):
        init.normal_(self.theta, mean=0, std=0.1)
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def scaled_laplacian(feat, D_invsqrt, graph):
            """计算缩放后的拉普拉斯矩阵 L' = L - I (定义域平移的关键)"""
            #  L = I - D^-1/2 A D^-1/2
            #  L' = L - I = -D^-1/2 A D^-1/2
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return -graph.ndata.pop('h') * D_invsqrt

        def chebyshev_basis(feat, L_prime, k):
            """计算调整后的切比雪夫基函数"""
            # T_0(L')x = x
            if k == 0:
                return feat
            
            # T_1(L')x = L'x
            if k == 1:
                return L_prime(feat)
            
            # T_k(L')x = 2*L'(T_{k-1}(L')x) - T_{k-2}(L')x
            T_k_minus_1 = L_prime(feat)
            T_k_minus_2 = feat
            for _ in range(2, k+1):
                T_k = 2 * L_prime(T_k_minus_1) - T_k_minus_2
                T_k_minus_2, T_k_minus_1 = T_k_minus_1, T_k
            return T_k

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            
            # ' = L - I
            L_prime = lambda x: scaled_laplacian(x, D_invsqrt, graph)
            
            h = self.theta[0] * feat  # T_0
            for order in range(1, self._k + 1):
                T_k = chebyshev_basis(feat, L_prime, order)
                
                #  (T_k + 1)/2
                adjusted_T_k = (T_k + 1) /2
                h += self.theta[order] * adjusted_T_k
            
            h = self.linear(h)
            if self.activation is not None:
                h = self.activation(h)
        return h


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, batch=False):
        super(BWGNN, self).__init__()
        self.g = graph
        self.conv = []
#         for i in range(len(self.thetas)):
#             if not batch:
#                 self.conv.append(PolyConv(h_feats, h_feats, self.thetas[i], lin=False))
#             else:
#                 self.conv.append(PolyConvBatch(h_feats, h_feats, self.thetas[i], lin=False))
        
        self.conv.append(DomainAdjustedChebyshevConv(h_feats,h_feats,0))
        self.conv.append(DomainAdjustedChebyshevConv(h_feats,h_feats,1))
        self.conv.append(DomainAdjustedChebyshevConv(h_feats,h_feats,3))
        
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats*len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.d = d

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0])
        for conv in self.conv:
            h0 = conv(self.g, h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h

    def testlarge(self, g, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0])
        for conv in self.conv:
            h0 = conv(g, h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h

    def batch(self, blocks, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)

        h_final = torch.zeros([len(in_feat),0])
        for conv in self.conv:
            h0 = conv(blocks[0], h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h