import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .utils.graph import Graph


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class CTRGCN_pretrain(nn.Module):
    def __init__(self, num_class=60, num_point=17, num_person=1, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, sigmoid=False):
        super(CTRGCN_pretrain, self).__init__()

        self.graph = Graph(**graph_args)
        A = self.graph.A
        assert A.shape == (3, 17, 17)
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel*4, num_class)
        self.sigmoid = sigmoid
        if self.sigmoid:
            self.activation_func = nn.Sigmoid()

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x_feature = self.drop_out(x)
        x = self.fc(x_feature)
        if self.sigmoid:
            x = self.activation_func(x)
        return x, x_feature


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


class CausalCTRGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(CausalCTRGCN, self).__init__()

        self.graph = Graph(**graph_args)
        A = self.graph.A
        assert A.shape == (3, 17, 17)
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        inter_channels = (base_channel*4) // 4
        self.inter_c = inter_channels
        self.sconv_a = nn.Conv2d(base_channel*4, inter_channels, 1)
        self.sconv_b = nn.Conv2d(base_channel*4, inter_channels, 1)
        self.tconv_a = nn.Conv2d(base_channel*4, inter_channels, 1)
        self.tconv_b = nn.Conv2d(base_channel*4, inter_channels, 1)
        self.soft = nn.Softmax(-2)
        # Causal head
        self.fcn_c = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fcn_c.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, label):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

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
        x_c = x_c.reshape(N, Cf, V, Tf).permute(0, 1, 3, 2).contiguous()

        x_c = x_c.view(N, M, Cf, -1)
        assert x_c.size(-1) == Tf * V
        x_feature_c = x_c.mean(3).mean(1)
        x_pred_c = self.fcn_c(x_feature_c)  # Causal prediction

        if label != None:
            A_st_nc = torch.ones_like(A_st_c) - A_st_c  # Non-causal graph
            # Construct counterfactual sample for each sample
            _, topk_indices = sample_max_label_diff_indices_vectorized(label.float())
            A_st_cf = A_st_c[topk_indices.view(-1)]  # Spatial–temporal counterfactual graph

            x_nc = torch.matmul(x, A_st_nc)  # Non-causal features
            x_nc = x_nc.reshape(N, Cf, V, Tf).permute(0, 1, 3, 2).contiguous()
            x_cf = torch.matmul(x, A_st_cf)  # Counterfactual features
            x_cf = x_cf.reshape(N, Cf, V, Tf).permute(0, 1, 3, 2).contiguous()
            x_nc = x_nc.view(N, M, Cf, -1)
            x_cf = x_cf.view(N, M, Cf, -1)
            x_feature_nc = x_nc.mean(3).mean(1)
            x_feature_cf = x_cf.mean(3).mean(1)
            x_feature_nc = self.drop_out(x_feature_nc)
            x_feature_cf = self.drop_out(x_feature_cf)
            x_pred_cf = self.fcn_c(x_feature_cf)  # Counterfactual prediction
            return x_pred_c, x_feature_nc, x_pred_cf
        else:
            return x_pred_c

class NonCausalHead(nn.Module):
    """Non-causal head used for finetuning."""
    def __init__(self, num_class, in_channels=256):
        super().__init__()
        # Non-causal head
        self.fcn_nc = nn.Linear(in_channels, num_class)

    def forward(self, x):
        x_pred_nc = self.fcn_nc(x)  # Non-causal prediction
        return x_pred_nc