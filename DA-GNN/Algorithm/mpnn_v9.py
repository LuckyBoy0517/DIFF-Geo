import torch
from torch import nn
from torch.autograd import Function
from dgl.nn.pytorch import GraphConv,NNConv
import os
import numpy as np
import dgl
import random as rd
import torch
import torch.nn as nn

class GradientReversalLayer(Function):

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


def grad_reverse(x, lambda_val=1.0):
    return GradientReversalLayer.apply(x, lambda_val)


class DomainClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lambda_val=1.0):
        x = grad_reverse(x, lambda_val)
        return self.classifier(x)


class MPNN(nn.Module):
    def __init__(self, aggregator_type,node_in_feats, node_hidden_dim, edge_input_dim, edge_hidden_dim,num_step_message_passing,gconv_dp,edge_dp,nn_dp1,
                 use_dann=False, domain_hidden_dim=64, domain_dp=0.3):

            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=edge_dp),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim),
            nn.Dropout(p=edge_dp))#1-4-32x32

        self.conv = NNConv(in_feats=node_hidden_dim,#32
               out_feats=node_hidden_dim,#32
               edge_func=edge_network,#32x32
               aggregator_type=aggregator_type)


        self.y_linear = nn.Linear(node_hidden_dim, 2)#4-4
        self.bn = nn.BatchNorm1d(node_hidden_dim)


        self.gnn_dropout = nn.Dropout(p=gconv_dp)#dropout
        self.nn_dropout = nn.Dropout(p=nn_dp1)

        #DANN
        if use_dann:
            self.domain_classifier = DomainClassifier(
                input_dim=node_hidden_dim,
                hidden_dim=domain_hidden_dim,
                dropout=domain_dp
            )

    def forward(self, g, n_feat, e_feat, lambda_val=1.0):
        out = torch.relu(self.lin0(n_feat))  # (B1, H1)

        for i in range(self.num_step_message_passing):
            out = torch.relu(self.conv(g, out, e_feat))
            out = self.gnn_dropout(out)# (B1, H1)

        y_bn = self.bn(out)
        y_sigmoid = torch.sigmoid(self.y_linear(y_bn))


        if self.use_dann:
            domain_pred = self.domain_classifier(y_bn.detach() if not self.training else y_bn,
                                                 lambda_val)
            return y_sigmoid, domain_pred
        return y_sigmoid
