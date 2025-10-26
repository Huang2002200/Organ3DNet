import sys

if sys.version_info[:2] >= (3, 8):
    from collections.abc import Sequence
else:
    from collections import Sequence

from enum import Enum

import torch.nn as nn
import MinkowskiEngine as ME
import torch


class SelfAttention1(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention1, self).__init__()
        self.w_qs = nn.Linear(in_channel, in_channel, bias=False)
        self.w_ks = nn.Linear(in_channel, in_channel, bias=False)
        self.w_vs = nn.Linear(in_channel, in_channel, bias=False)
        self.fc_gamma = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, in_channel)
        )
        self.fc_delta = nn.Sequential(
            nn.Linear(4, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, in_channel))
        self.CONV1 = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, 1),
            nn.BatchNorm1d(in_channel),
            nn.ReLU())
        self.CONV2 = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, 1),
            nn.BatchNorm1d(in_channel),
            nn.ReLU())

    def forward(self, pose_enc, knn_idx, new_points, k=16, d=1):
        pos_enc = self.fc_delta(pose_enc)  # b x n x k x f
        x = self.CONV2(new_points.transpose(1, 2)).transpose(1, 2)
        # q:bnf,k\v:bnkf
        q, k, v = self.w_qs(x), torch_util.index_points(self.w_ks(x), knn_idx), torch_util.index_points(self.w_vs(x), knn_idx)
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)  # q[:, :, None]:b x n x 1 x f
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.CONV1(res.transpose(1, 2)).transpose(1, 2) + new_points#bnc
        return res

def dg_knn(adj_matrix, k=20, d=3):
    """Get KNN based on the pairwise distance.
    Args:
    adj_matrix: (num_points, num_points)
    k: int

    Returns:
    nearest neighbors: (num_points, k)
    """
    k1 = k * d + 1
    neg_adj = -adj_matrix
    _, nn_idx = torch.topk(neg_adj, k=k1)

    nn_idx = nn_idx[:, 1:k1:d]
    return nn_idx

# def pairwise_distance(point_cloud):
#     """Compute pairwise distance of a point cloud.
#     Args:
#       point_cloud: tensor (num_points, num_dims)
#     Returns:
#       pairwise distance: (num_points, num_points)
#       (i,j)对应点云内部第i个点与第 j 个点之间的距离
#     """
#     point_cloud = point_cloud.features
#     point_cloud_inner = torch.matmul(point_cloud, point_cloud.transpose(0, 1))
#     point_cloud_inner = -2 * point_cloud_inner
#     point_cloud_square = torch.sum(torch.square(point_cloud), dim=-1, keepdim=True)
#     point_cloud_square_transpose = point_cloud_square.permute(1, 0)
#     return point_cloud_square + point_cloud_inner + point_cloud_square_transpose

def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.
    Args:
      point_cloud: tensor (num_points, num_dims)
    Returns:
      pairwise distance: (num_points, num_points)
      (i,j)对应点云内部第i个点与第 j 个点之间的距离
    """
    point_cloud_inner = torch.matmul(point_cloud, point_cloud.transpose(0, 1))
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = torch.sum(torch.square(point_cloud), dim=-1, keepdim=True)
    point_cloud_square_transpose = point_cloud_square.permute(1, 0)
    return point_cloud_square + point_cloud_inner + point_cloud_square_transpose


def dg_knn(point_cloud, k=20, d=3):
    """Get KNN based on the pairwise distance.
    Args:
    adj_matrix: (num_points, num_points)
    k: int

    Returns:
    nearest neighbors: (num_points, k)
    """
    adj_matrix = pairwise_distance(point_cloud)
    k1 = k * d + 1
    neg_adj = -adj_matrix
    _, nn_idx = torch.topk(neg_adj, k=k1)

    nn_idx = nn_idx[:, 1:k1:d]
    return nn_idx


def get_edge_feature(features, nn_idx, k=20):
    """
    Construct edge features for each point in a sparse tensor.

    Args:
        point_cloud: ME.SparseTensor
        nn_idx: (num_points, k)  # Indices of k-nearest neighbors
        k: int

    Returns:
        edge_features: (num_points, k, 2*num_dims)  # tensor
    """
    # Extract coordinates and features from the sparse tensor


    # Gather neighbor features
    point_cloud_neighbors = features[nn_idx]  # (num_points, k, num_dims)

    # Expand central features and compute edge features
    point_cloud_central = features.unsqueeze(dim=-2)  # (num_points, 1, num_dims)
    point_cloud_central = point_cloud_central.expand(-1, k, -1)  # (num_points, k, num_dims)

    # Compute the edge features as concatenation of central features and difference with neighbors
    edge_feature = torch.cat([point_cloud_central, point_cloud_neighbors - point_cloud_central], dim=-1)

    return edge_feature


class NormType(Enum):
    BATCH_NORM = 0
    INSTANCE_NORM = 1
    INSTANCE_BATCH_NORM = 2


def get_norm(norm_type, n_channels, D, bn_momentum=0.1):
    if norm_type == NormType.BATCH_NORM:
        return ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
    elif norm_type == NormType.INSTANCE_NORM:
        return ME.MinkowskiInstanceNorm(n_channels)
    elif norm_type == NormType.INSTANCE_BATCH_NORM:
        return nn.Sequential(
            ME.MinkowskiInstanceNorm(n_channels),
            ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum),
        )
    else:
        raise ValueError(f"Norm type: {norm_type} not supported")


class ConvType(Enum):
    """
    Define the kernel region type
    """

    HYPERCUBE = 0, "HYPERCUBE"
    SPATIAL_HYPERCUBE = 1, "SPATIAL_HYPERCUBE"
    SPATIO_TEMPORAL_HYPERCUBE = 2, "SPATIO_TEMPORAL_HYPERCUBE"
    HYPERCROSS = 3, "HYPERCROSS"
    SPATIAL_HYPERCROSS = 4, "SPATIAL_HYPERCROSS"
    SPATIO_TEMPORAL_HYPERCROSS = 5, "SPATIO_TEMPORAL_HYPERCROSS"
    SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS = (
        6,
        "SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS ",
    )

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


# Convert the ConvType var to a RegionType var
conv_to_region_type = {
    # kernel_size = [k, k, k, 1]
    ConvType.HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIAL_HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIO_TEMPORAL_HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIO_TEMPORAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS: ME.RegionType.HYPER_CUBE,  # JONAS CHANGE from HYBRID
}

# int_to_region_type = {m.value: m for m in ME.RegionType}
int_to_region_type = {m: ME.RegionType(m) for m in range(3)}


def convert_region_type(region_type):
    """
    Convert the integer region_type to the corresponding RegionType enum object.
    """
    return int_to_region_type[region_type]


def convert_conv_type(conv_type, kernel_size, D):
    assert isinstance(conv_type, ConvType), "conv_type must be of ConvType"
    region_type = conv_to_region_type[conv_type]
    axis_types = None
    if conv_type == ConvType.SPATIAL_HYPERCUBE:
        # No temporal convolution
        if isinstance(kernel_size, Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [
                kernel_size,
            ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCUBE:
        # conv_type conversion already handled
        assert D == 4
    elif conv_type == ConvType.HYPERCUBE:
        # conv_type conversion already handled
        pass
    elif conv_type == ConvType.SPATIAL_HYPERCROSS:
        if isinstance(kernel_size, Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [
                kernel_size,
            ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.HYPERCROSS:
        # conv_type conversion already handled
        pass
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCROSS:
        # conv_type conversion already handled
        assert D == 4
    elif conv_type == ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS:
        # Define the CUBIC conv kernel for spatial dims and CROSS conv for temp dim
        axis_types = [
            ME.RegionType.HYPER_CUBE,
        ] * 3
        if D == 4:
            axis_types.append(ME.RegionType.HYPER_CROSS)
    return region_type, axis_types, kernel_size


def conv(
    in_planes,
    out_planes,
    kernel_size,
    stride=1,
    dilation=1,
    bias=False,
    conv_type=ConvType.HYPERCUBE,
    D=-1,
):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(
        conv_type, kernel_size, D
    )
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        region_type=region_type,
        axis_types=None,  # axis_types JONAS
        dimension=D,
    )

    return ME.MinkowskiConvolution(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        bias=bias,
        kernel_generator=kernel_generator,
        dimension=D,
    )


def conv_tr(
    in_planes,
    out_planes,
    kernel_size,
    upsample_stride=1,
    dilation=1,
    bias=False,
    conv_type=ConvType.HYPERCUBE,
    D=-1,
):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(
        conv_type, kernel_size, D
    )
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        upsample_stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D,
    )

    return ME.MinkowskiConvolutionTranspose(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=upsample_stride,
        dilation=dilation,
        bias=bias,
        kernel_generator=kernel_generator,
        dimension=D,
    )


def avg_pool(
    kernel_size,
    stride=1,
    dilation=1,
    conv_type=ConvType.HYPERCUBE,
    in_coords_key=None,
    D=-1,
):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(
        conv_type, kernel_size, D
    )
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D,
    )

    return ME.MinkowskiAvgPooling(
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        kernel_generator=kernel_generator,
        dimension=D,
    )


def avg_unpool(
    kernel_size, stride=1, dilation=1, conv_type=ConvType.HYPERCUBE, D=-1
):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(
        conv_type, kernel_size, D
    )
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D,
    )

    return ME.MinkowskiAvgUnpooling(
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        kernel_generator=kernel_generator,
        dimension=D,
    )


def sum_pool(
    kernel_size, stride=1, dilation=1, conv_type=ConvType.HYPERCUBE, D=-1
):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(
        conv_type, kernel_size, D
    )
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D,
    )

    return ME.MinkowskiSumPooling(
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        kernel_generator=kernel_generator,
        dimension=D,
    )
