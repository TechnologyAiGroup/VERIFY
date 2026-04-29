import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np
from torch import nn
from torch.nn import init
import math

class BernsteinConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k,  
                 n=2,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(BernsteinConv, self).__init__()
        self._k = k
        self._n = n
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias) if lin else None
        self.lin = lin
        self.comb = math.comb(n, k)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            
            h = feat.clone()
            
            if self._k == 0:  # (1-x)^n
                for _ in range(self._n):
                    h = unnLaplacian(h, D_invsqrt, graph) / 2  # x = L/2
                    h = feat - h  # I - x
            elif self._k == self._n:  # x^n
                for _ in range(self._n):
                    h = unnLaplacian(h, D_invsqrt, graph) / 2  # x = L/2
            else:  # x^k * (1-x)^{n-k}
                temp = feat.clone()
                for _ in range(self._k):
                    temp = unnLaplacian(temp, D_invsqrt, graph) / 2  # x = L/2
                h_part = feat.clone()
                for _ in range(self._n - self._k):
                    h_part = unnLaplacian(h_part, D_invsqrt, graph) / 2  # x = L/2
                    h_part = feat - h_part  # I - x
                h = temp * h_part
            
            h = self.comb * h

        if self.lin and self.linear is not None:
            h = self.linear(h)
            h = self.activation(h)
        return h


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, batch=False):
        super(BWGNN, self).__init__()
        self.g = graph
        self.n = 2

        self.conv = nn.ModuleList([
            BernsteinConv(h_feats, h_feats, k=0, n=2, lin=False),
            BernsteinConv(h_feats, h_feats, k=1, n=2, lin=False),
            BernsteinConv(h_feats, h_feats, k=2, n=2, lin=False)
        ])
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        
        h_final = torch.cat([conv(self.g, h) for conv in self.conv], dim=-1)
        
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h

    def testlarge(self, g, in_feat):
        return self.forward(in_feat)

    def batch(self, blocks, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        
        h_final = torch.cat([conv(blocks[0], h) for conv in self.conv], dim=-1)
        
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h