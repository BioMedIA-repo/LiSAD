import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class DenseLayer(nn.Module):
    def __init__(self, c_in, c_out, zero_init=False):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out)

        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            bound = np.sqrt(6 / (c_in + c_out))
            nn.init.uniform_(self.linear.weight.data, -bound, bound)

        nn.init.zeros_(self.linear.bias.data)

    def forward(self, node_feats):
        return self.linear(node_feats)


class SineLayer(nn.Module):
    def __init__(self, c_in, c_out, bias=True, zero_init=False, omega_0=1):
        super().__init__()
        self.omega_0 = omega_0
        self.zero_init = zero_init
        self.in_features = c_in
        self.linear = nn.Linear(c_in, c_out, bias=bias)

        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            bound = np.sqrt(6 / (c_in + c_out))
            nn.init.uniform_(self.linear.weight.data, -bound, bound)

        if bias:
            nn.init.zeros_(self.linear.bias.data)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class HGNNConv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super().__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))

        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)

        if self.bias is not None:
            x = x + self.bias

        x = G.matmul(x)
        return x


class HGNNClassifier(nn.Module):
    def __init__(
        self,
        spatial_dim,
        n_genes,
        n_hid,
        n_embed,
        n_class,
        omega_0=1.0,
        dropout=0.3,
    ):
        super().__init__()
        self.dropout = dropout

        mid_channel = 200
        bottleneck_dim = 30

        self.inr = nn.Sequential(
            SineLayer(spatial_dim, mid_channel, omega_0=omega_0),
            SineLayer(mid_channel, mid_channel, omega_0=omega_0),
            SineLayer(mid_channel, bottleneck_dim, omega_0=omega_0),
            DenseLayer(bottleneck_dim, n_genes),
        )

        self.feature_projector = nn.Sequential(
            nn.Linear(n_genes, n_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.hgc1 = HGNNConv(n_hid, n_hid)
        self.hgc2 = HGNNConv(n_hid, n_embed)

        self.bn_embed = nn.BatchNorm1d(n_embed)
        self.classifier = nn.Linear(n_embed, n_class)

    def forward(self, coords, G):
        x_rec = self.inr(coords)
        x_feat = self.feature_projector(x_rec)

        x = self.hgc1(x_feat, G)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        latent = self.hgc2(x, G)

        h = self.bn_embed(latent)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)

        logits = self.classifier(h)

        return logits, x_rec, latent