import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.tgcn import ConvTemporalGraphical
from .utils.graph import Graph
import math


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def sample_max_label_diff_indices_vectorized(labels):
    """Sampling of indices with maximum label difference for each sample."""
    diff = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))  # # Compute the label difference matrix (N, N)
    diff.fill_diagonal_(-1)  # Avoid self-pairs
    # Get the maximum value of each row for constructing a mask
    max_vals = diff.max(dim=1, keepdim=True).values  # (N, 1)
    if (max_vals == 0).all():
        raise ValueError(f"{max_vals} contains all zeros.")
    # Construct a mask: positions where the value equals the maximum label difference
    max_mask = (diff == max_vals)  # (N, N), boolean mask
    # For subsequent sampling, convert the boolean mask to float and multiply by a random number.
    # Positions in the mask get a random value in [0,1), others are set to -1
    rand = torch.rand_like(diff)  # Create a random tensor `rand` with the same shape as `diff`, with elements uniformly distributed in [0,1)
    masked_rand = torch.where(max_mask, rand, torch.full_like(diff, -1.0))
    # For positions where max_mask == True, use the corresponding values from rand
    # For positions where max_mask == False, fill with -1.0
    topk_values, topk_indices = torch.topk(masked_rand, k=1, dim=1)  # Select top-1 per row
    return topk_values, topk_indices  # Each row holds index randomly drawn from the positions with the maximum label difference for the current sample


class BaseModel(nn.Module):
    """Base ST-GCN model used for pretraining."""
    def __init__(self, in_channels, num_class, graph_args, sigmoid=False, **kwargs):
        super().__init__()
        # Load adjacency matrix from graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)  # Fixed adjacency matrix
        spatial_kernel_size = A.size(0)
        spatial_num_nodes = A.size(1)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size, spatial_num_nodes)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        # Build ST-GCN layers
        self.st_gcn_networks = nn.ModuleList((
            base_st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            base_st_gcn(64, 64, kernel_size, 1, **kwargs),
            base_st_gcn(64, 64, kernel_size, 1, **kwargs),
            base_st_gcn(64, 64, kernel_size, 1, **kwargs),
            base_st_gcn(64, 128, kernel_size, 2, **kwargs),
            base_st_gcn(128, 128, kernel_size, 1, **kwargs),
            base_st_gcn(128, 128, kernel_size, 1, **kwargs),
            base_st_gcn(128, 256, kernel_size, 2, **kwargs),
            base_st_gcn(256, 256, kernel_size, 1, **kwargs),
            base_st_gcn(256, 256, kernel_size, 1, **kwargs)
        ))
        # Assessment head
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        self.sigmoid = sigmoid
        if self.sigmoid:
            self.activation_func = nn.Sigmoid()

    def forward(self, x):
        N, C, T, V, M = x.size()
        # Reshape and normalize input
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # Apply ST-GCN layers
        for gcn in self.st_gcn_networks:
            x = gcn(x, self.A)

        # Global average pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x_feature = x.view(N, M, -1, 1, 1).mean(dim=1)
        # Final prediction
        x = self.fcn(x_feature)
        x = x.view(x.size(0), -1)
        if self.sigmoid:
            x = self.activation_func(x)
        return x, x_feature.view(x.size(0), -1)


class CausalModel(nn.Module):
    """Main model used for finetuning."""
    def __init__(self, in_channels, num_class, graph_args, coff_embedding=4, **kwargs):
        super().__init__()
        # Load adjacency matrix from graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)  # Fixed adjacency matrix
        spatial_kernel_size = A.size(0)
        spatial_num_nodes = A.size(1)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size, spatial_num_nodes)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        # Build ST-GCN layers
        self.st_gcn_networks = nn.ModuleList((
            base_st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            base_st_gcn(64, 64, kernel_size, 1, **kwargs),
            base_st_gcn(64, 64, kernel_size, 1, **kwargs),
            base_st_gcn(64, 64, kernel_size, 1, **kwargs),
            base_st_gcn(64, 128, kernel_size, 2, **kwargs),
            base_st_gcn(128, 128, kernel_size, 1, **kwargs),
            base_st_gcn(128, 128, kernel_size, 1, **kwargs),
            base_st_gcn(128, 256, kernel_size, 2, **kwargs),
            base_st_gcn(256, 256, kernel_size, 1, **kwargs),
            base_st_gcn(256, 256, kernel_size, 1, **kwargs)
        ))

        inter_channels = (256 * kernel_size[1]) // coff_embedding
        self.inter_c = inter_channels
        self.sconv_a = nn.Conv2d(256, inter_channels, 1)
        self.sconv_b = nn.Conv2d(256, inter_channels, 1)
        self.tconv_a = nn.Conv2d(256, inter_channels, 1)
        self.tconv_b = nn.Conv2d(256, inter_channels, 1)
        self.soft = nn.Softmax(-2)
        # Causal head
        self.fcn_c = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x, label):
        N, C, T, V, M = x.size()
        # Reshape and normalize input
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # Apply ST-GCN layers
        for gcn in self.st_gcn_networks:
            x = gcn(x, self.A)

        # Compute spatial–temporal dependency graph
        _, Cf, Tf, _ = x.size()
        sx1 = self.sconv_a(x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * Tf)
        sx2 = self.sconv_b(x).view(N, self.inter_c * Tf, V)
        As_c = self.soft(torch.matmul(sx1, sx2) / math.sqrt(sx1.size(-1)))
        tx1 = self.tconv_a(x).permute(0, 2, 1, 3).contiguous().view(N, Tf, self.inter_c * V)
        tx2 = self.tconv_b(x).permute(0, 1, 3, 2).contiguous().view(N, self.inter_c * V, Tf)
        At_c = self.soft(torch.matmul(tx1, tx2) / math.sqrt(tx1.size(-1)))
        As_c_exp = As_c.unsqueeze(2).unsqueeze(4)
        At_c_exp = At_c.unsqueeze(1).unsqueeze(3)
        A_st_c = (As_c_exp * At_c_exp).reshape(N, V * Tf, V * Tf)  # Causal graph
        x = x.permute(0, 1, 3, 2).reshape(N, Cf, V * Tf)
        x_c = torch.matmul(x, A_st_c)  # Causal features
        x_c = x_c.reshape(N, Cf, V, Tf).permute(0, 1, 3, 2)
        x_c = F.avg_pool2d(x_c, x_c.size()[2:])
        x_feature_c = x_c.view(N, M, -1, 1, 1).mean(dim=1)
        x_pred_c = self.fcn_c(x_feature_c)  # Causal prediction
        x_pred_c = x_pred_c.view(x_pred_c.size(0), -1)

        if label != None:
            A_st_nc = torch.ones_like(A_st_c) - A_st_c  # Non-causal graph
            # Construct counterfactual sample for each sample
            _, topk_indices = sample_max_label_diff_indices_vectorized(label.float())
            A_st_cf = A_st_c[topk_indices.view(-1)]  # Spatial–temporal counterfactual graph

            x_nc = torch.matmul(x, A_st_nc)  # Non-causal features
            x_nc = x_nc.reshape(N, Cf, V, Tf).permute(0, 1, 3, 2)
            x_cf = torch.matmul(x, A_st_cf)  # Counterfactual features
            x_cf = x_cf.reshape(N, Cf, V, Tf).permute(0, 1, 3, 2)
            x_nc = F.avg_pool2d(x_nc, x_nc.size()[2:])
            x_cf = F.avg_pool2d(x_cf, x_cf.size()[2:])
            x_feature_nc = x_nc.view(N, M, -1, 1, 1).mean(dim=1)
            x_feature_cf = x_cf.view(N, M, -1, 1, 1).mean(dim=1)
            x_pred_cf = self.fcn_c(x_feature_cf).view(N, -1)  # Counterfactual prediction
            return x_pred_c, x_feature_nc, x_pred_cf
        else:
            return x_pred_c


class NonCausalHead(nn.Module):
    """Non-causal head used for finetuning."""
    def __init__(self, num_class, in_channels=256):
        super().__init__()
        # Non-causal head
        self.fcn_nc = nn.Conv2d(in_channels, num_class, kernel_size=1)

    def forward(self, x):
        x_pred_nc = self.fcn_nc(x)  # Non-causal prediction
        x_pred_nc = x_pred_nc.view(x_pred_nc.size(0), -1)
        return x_pred_nc


class base_st_gcn(nn.Module):
    """Define ST-GCN layer."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.5, residual=True):
        super().__init__()
        assert len(kernel_size) == 3
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        # Spatial graph convolution
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.edge_importance = nn.Parameter(torch.zeros(kernel_size[1], kernel_size[2], kernel_size[2]))
        nn.init.constant_(self.edge_importance, 1e-6)
        # Temporal convolution
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A + self.edge_importance)  # Spatial graph convolution
        x = self.tcn(x) + res  # Temporal convolution
        return self.relu(x)




