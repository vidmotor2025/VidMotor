import math
from functools import partial
from einops import rearrange
import torch
import torch_dct as dct
import torch.nn as nn
from timm.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FreqMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b, f, _ = x.shape
        x = dct.dct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = dct.idct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MixedBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)
        self.mlp2 = FreqMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        b, f, c = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x1 = x[:, :f // 2] + self.drop_path(self.mlp1(self.norm2(x[:, :f // 2])))
        x2 = x[:, f // 2:] + self.drop_path(self.mlp2(self.norm3(x[:, f // 2:])))
        return torch.cat((x1, x2), dim=1)


class PoseTransformerV2(nn.Module):
    def __init__(self, num_joints=17, in_chans=2, embed_dim_ratio=2, depth=1,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None,
                 number_of_kept_frames=1, number_of_kept_coeffs=1):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3
        self.num_frame_kept = number_of_kept_frames
        self.num_coeff_kept = number_of_kept_coeffs

        ### spatial patch embedding
        self.Joint_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Freq_embedding = nn.Linear(in_chans * num_joints, embed_dim)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_frame_kept, embed_dim))
        self.Temporal_pos_embed_ = nn.Parameter(torch.zeros(1, self.num_coeff_kept, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            MixedBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=self.num_coeff_kept, out_channels=1, kernel_size=1)
        self.weighted_mean_ = torch.nn.Conv1d(in_channels=self.num_frame_kept, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, out_dim),
        )

    def Spatial_forward_features(self, x):
        b, f, p, _ = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        num_frame_kept = self.num_frame_kept
        index = torch.arange((f - 1) // 2 - num_frame_kept // 2, (f - 1) // 2 + num_frame_kept // 2 + 1)
        x = self.Joint_embedding(x[:, index].view(b * num_frame_kept, p, -1))
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) p c -> b f (p c)', f=num_frame_kept)
        return x

    def forward_features(self, x, Spatial_feature):
        b, f, p, _ = x.shape
        num_coeff_kept = self.num_coeff_kept
        x = dct.dct(x.permute(0, 2, 3, 1))[:, :, :, :num_coeff_kept]
        x = x.permute(0, 3, 1, 2).contiguous().view(b, num_coeff_kept, -1)
        x = self.Freq_embedding(x)
        Spatial_feature += self.Temporal_pos_embed
        x += self.Temporal_pos_embed_
        x = torch.cat((x, Spatial_feature), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.Temporal_norm(x)
        return x

    def forward(self, x, return_rep=True):
        b, f, p, _ = x.shape
        x_ = x.clone()
        Spatial_feature = self.Spatial_forward_features(x)
        x = self.forward_features(x_, Spatial_feature)
        x = torch.cat((self.weighted_mean(x[:, :self.num_coeff_kept]), self.weighted_mean_(x[:, self.num_coeff_kept:])), dim=-1)
        if return_rep:
            return x
        x = self.head(x).view(b, 1, p, -1)
        return x


class ClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes=3, classifier_dropout=0.5, classifier_hidden_dims=2048):
        super(ClassifierHead, self).__init__()
        self.dims = [input_dim, classifier_hidden_dims, num_classes]
        self.fc_layers = self._create_fc_layers()
        self.batch_norms = self._create_batch_norms()
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.activation = nn.ReLU()

    def _create_fc_layers(self):
        fc_layers = nn.ModuleList()
        mlp_size = len(self.dims)
        for i in range(mlp_size - 1):
            fc_layer = nn.Linear(in_features=self.dims[i],
                                 out_features=self.dims[i + 1])
            fc_layers.append(fc_layer)
        return fc_layers

    def _create_batch_norms(self):
        batch_norms = nn.ModuleList()
        n_batchnorms = len(self.dims) - 2
        if n_batchnorms == 0:
            return batch_norms
        for i in range(n_batchnorms):
            batch_norm = nn.BatchNorm1d(self.dims[i + 1], momentum=0.1)
            batch_norms.append(batch_norm)
        return batch_norms

    def forward(self, feat):
        feat = self.dropout(feat)
        B, _, C = feat.shape
        assert feat.shape[1] == 1
        feat = feat.reshape(B, C)  # (B, 1, C) -> (B, C)
        return self._forward_poseformerv2(feat), feat

    def _forward_fc_layers(self, feat):
        mlp_size = len(self.dims)
        for i in range(mlp_size - 2):
            fc_layer = self.fc_layers[i]
            batch_norm = self.batch_norms[i]
            feat = self.activation(batch_norm(fc_layer(feat)))
        last_fc_layer = self.fc_layers[-1]
        feat = last_fc_layer(feat)
        return feat

    def _forward_poseformerv2(self, feat):
        """
        x: Tensor with shape (batch_size, 1, embed_dim_ratio * num_joints * 2)
        """
        feat = self._forward_fc_layers(feat)
        return feat


class Model_pretrain(nn.Module):
    def __init__(
        self,
        in_channels,
        num_class,
        num_joints=17,
        embed_dim_ratio=32,
        depth=4,
        num_heads=8,
        number_of_kept_frames=27,
        number_of_kept_coeffs=27,
        drop_path_rate=0.1,
        classifier_dropout=0.5,
        classifier_hidden_dims=2048,
        sigmoid=True
    ):
        super().__init__()

        self.backbone = PoseTransformerV2(
            num_joints=num_joints,
            in_chans=in_channels,
            embed_dim_ratio=embed_dim_ratio,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=2,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            number_of_kept_frames=number_of_kept_frames,
            number_of_kept_coeffs=number_of_kept_coeffs,
        )
        self.feature_dim = embed_dim_ratio * num_joints * 2
        self.head = ClassifierHead(self.feature_dim, num_classes=num_class, classifier_dropout=classifier_dropout, classifier_hidden_dims=classifier_hidden_dims)
        self.sigmoid = sigmoid
        if self.sigmoid:
            self.activation_func = nn.Sigmoid()

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 2, 3, 1).contiguous().view(N * M, T, V, C)
        feat = self.backbone(x, return_rep=True)
        out, out_feat = self.head(feat)
        if self.sigmoid:
            out = self.activation_func(out)
        return out, out_feat


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


class PoseTransformerV2_finetune(nn.Module):
    def __init__(self, num_joints=17, in_chans=2, embed_dim_ratio=2, depth=1,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None,
                 number_of_kept_frames=1, number_of_kept_coeffs=1):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3
        self.num_frame_kept = number_of_kept_frames
        self.num_coeff_kept = number_of_kept_coeffs

        ### spatial patch embedding
        self.Joint_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Freq_embedding = nn.Linear(in_chans * num_joints, embed_dim)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_frame_kept, embed_dim))
        self.Temporal_pos_embed_ = nn.Parameter(torch.zeros(1, self.num_coeff_kept, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            MixedBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=self.num_coeff_kept, out_channels=1, kernel_size=1)
        self.weighted_mean_ = torch.nn.Conv1d(in_channels=self.num_frame_kept, out_channels=1, kernel_size=1)

        inter_channels = embed_dim // 4
        self.stconv_a = nn.Conv2d(embed_dim, inter_channels, 1)
        self.stconv_b = nn.Conv2d(embed_dim, inter_channels, 1)
        self.soft = nn.Softmax(-2)

    def Spatial_forward_features(self, x):
        b, f, p, _ = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        num_frame_kept = self.num_frame_kept

        index = torch.arange((f - 1) // 2 - num_frame_kept // 2, (f - 1) // 2 + num_frame_kept // 2 + 1)
        x = self.Joint_embedding(x[:, index].view(b * num_frame_kept, p, -1))  # x: [4, 40, 17, 3]; input: [108, 17, 3] -> [108, 17, 32]
        x += self.Spatial_pos_embed  # [108, 17, 32]
        x = self.pos_drop(x)  # [108, 17, 32]
        for blk in self.Spatial_blocks:
            x = blk(x)  # [108, 17, 32]
        x = self.Spatial_norm(x)  # [108, 17, 32]
        x = rearrange(x, '(b f) p c -> b f (p c)', f=num_frame_kept)
        return x

    def forward_features(self, x, Spatial_feature):
        b, f, p, _ = x.shape
        num_coeff_kept = self.num_coeff_kept
        x = dct.dct(x.permute(0, 2, 3, 1))[:, :, :, :num_coeff_kept]
        x = x.permute(0, 3, 1, 2).contiguous().view(b, num_coeff_kept, -1)
        x = self.Freq_embedding(x)
        Spatial_feature += self.Temporal_pos_embed
        x += self.Temporal_pos_embed_
        x = torch.cat((x, Spatial_feature), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.Temporal_norm(x)
        return x

    def forward(self, x, label):
        b, f, p, _ = x.shape
        x_ = x.clone()  # [4, 40, 17, 3], N T V C
        Spatial_feature = self.Spatial_forward_features(x)  # [4, 27, 544]
        x_tokens = self.forward_features(x_, Spatial_feature)  # [4, 54, 544]

        # Compute spatial–temporal dependency graph
        x_tokens = x_tokens.permute(0, 2, 1).unsqueeze(-1)  # [4, 544, 54, 1]
        stx1 = self.stconv_a(x_tokens).permute(0, 2, 1, 3).squeeze(-1)  # N, VTf, Cf
        stx2 = self.stconv_b(x_tokens).squeeze(-1)  # N, Cf, VTf
        x_tokens = x_tokens.squeeze(-1)  # [4, 544, 54]
        A_st_c = self.soft(torch.matmul(stx1, stx2) / math.sqrt(stx1.size(-1)))  # N, VTf, VTf; Causal graph
        x_c = torch.matmul(x_tokens, A_st_c).permute(0, 2, 1)  # N, Cf, VTf -> N, VTf, Cf; Causal features
        x_feature_c = torch.cat((self.weighted_mean(x_c[:, :self.num_coeff_kept]), self.weighted_mean_(x_c[:, self.num_coeff_kept:])), dim=-1)  # [4, 1, 544], [4, 1, 544] -> [4, 1, 1088]
        # stx1.shape, stx2.shape, A_st_c.shape, x_c.shape, x_feature_c.shape
        # [4, 54, 136], [4, 136, 54], [4, 54, 54], [4, 54, 544], [4, 1, 1088]

        if label != None:
            A_st_nc = torch.ones_like(A_st_c) - A_st_c  # Non-causal graph
            # Construct counterfactual sample for each sample
            _, topk_indices = sample_max_label_diff_indices_vectorized(label.float())
            A_st_cf = A_st_c[topk_indices.view(-1)]  # Spatial–temporal counterfactual graph

            x_nc = torch.matmul(x_tokens, A_st_nc).permute(0, 2, 1)  # Non-causal features
            x_cf = torch.matmul(x_tokens, A_st_cf).permute(0, 2, 1)  # Counterfactual features

            x_feature_nc = torch.cat((self.weighted_mean(x_nc[:, :self.num_coeff_kept]), self.weighted_mean_(x_nc[:, self.num_coeff_kept:])), dim=-1)
            x_feature_cf = torch.cat((self.weighted_mean(x_cf[:, :self.num_coeff_kept]), self.weighted_mean_(x_cf[:, self.num_coeff_kept:])), dim=-1)
            return x_feature_c, x_feature_nc, x_feature_cf
        else:
            return x_feature_c


class ClassifierHead_finetune(nn.Module):
    def __init__(self, input_dim, num_classes=3, classifier_dropout=0.5, classifier_hidden_dims=2048):
        super(ClassifierHead_finetune, self).__init__()
        self.dims = [input_dim, classifier_hidden_dims, num_classes]
        self.fc_layers = self._create_fc_layers()
        self.batch_norms = self._create_batch_norms()
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.activation = nn.ReLU()

    def _create_fc_layers(self):
        fc_layers = nn.ModuleList()
        mlp_size = len(self.dims)
        for i in range(mlp_size - 1):
            fc_layer = nn.Linear(in_features=self.dims[i],
                                 out_features=self.dims[i + 1])
            fc_layers.append(fc_layer)
        return fc_layers

    def _create_batch_norms(self):
        batch_norms = nn.ModuleList()
        n_batchnorms = len(self.dims) - 2
        if n_batchnorms == 0:
            return batch_norms
        for i in range(n_batchnorms):
            batch_norm = nn.BatchNorm1d(self.dims[i + 1], momentum=0.1)
            batch_norms.append(batch_norm)
        return batch_norms

    def forward(self, feat):
        feat = self.dropout(feat)
        B, _, C = feat.shape
        assert feat.shape[1] == 1
        feat = feat.reshape(B, C)  # (B, 1, C) -> (B, C)
        return self._forward_poseformerv2(feat)

    def _forward_fc_layers(self, feat):
        mlp_size = len(self.dims)
        for i in range(mlp_size - 2):
            fc_layer = self.fc_layers[i]
            batch_norm = self.batch_norms[i]
            feat = self.activation(batch_norm(fc_layer(feat)))
        last_fc_layer = self.fc_layers[-1]
        feat = last_fc_layer(feat)
        return feat

    def _forward_poseformerv2(self, feat):
        """
        x: Tensor with shape (batch_size, 1, embed_dim_ratio * num_joints * 2)
        """
        feat = self._forward_fc_layers(feat)
        return feat


class Model_finetune(nn.Module):
    def __init__(
        self,
        in_channels,
        num_class,
        num_joints=17,
        embed_dim_ratio=32,
        depth=4,
        num_heads=8,
        number_of_kept_frames=27,
        number_of_kept_coeffs=27,
        drop_path_rate=0.1,
        classifier_dropout=0.5,
        classifier_hidden_dims=2048
    ):
        super().__init__()

        self.backbone = PoseTransformerV2_finetune(
            num_joints=num_joints,
            in_chans=in_channels,
            embed_dim_ratio=embed_dim_ratio,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=2,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            number_of_kept_frames=number_of_kept_frames,
            number_of_kept_coeffs=number_of_kept_coeffs,
        )
        self.feature_dim = embed_dim_ratio * num_joints * 2
        self.head = ClassifierHead_finetune(self.feature_dim, num_classes=num_class, classifier_dropout=classifier_dropout, classifier_hidden_dims=classifier_hidden_dims)

    def forward(self, x, label):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 2, 3, 1).contiguous().view(N * M, T, V, C)
        if label != None:
            x_feature_c, x_feature_nc, x_feature_cf = self.backbone(x, label)
            x_pred_c = self.head(x_feature_c)
            x_pred_cf = self.head(x_feature_cf)
            return x_pred_c, x_feature_nc, x_pred_cf
        else:
            x_feature_c = self.backbone(x, label)
            x_pred_c = self.head(x_feature_c)
            return x_pred_c


class NonCausalHead(nn.Module):
    def __init__(
        self,
        num_class,
        num_joints=17,
        embed_dim_ratio=32,
        classifier_dropout=0.5,
        classifier_hidden_dims=2048
    ):
        super().__init__()
        self.feature_dim = embed_dim_ratio * num_joints * 2
        self.head = ClassifierHead_finetune(self.feature_dim, num_classes=num_class, classifier_dropout=classifier_dropout, classifier_hidden_dims=classifier_hidden_dims)

    def forward(self, feat):
        x_pred_nc = self.head(feat)
        return x_pred_nc
