from third_party.pointnet2.pointnet2_utils import furthest_point_sample
import torch

def random_sample(index, num_samples):
    if len(index) <= num_samples:
        return index
    return torch.randperm(len(index))[:num_samples]

def sample_3DEPS(coords, is_edge, num_queries):

    is_edge = is_edge.squeeze(-1)
    device = coords.device

    num_boundary = int(num_queries * 0.5)
    boundary_indices = (is_edge == 1).nonzero(as_tuple=True)[1]
    boundary_coords = coords[0, boundary_indices]

    sampled_boundary_indices = furthest_point_sample(boundary_coords.unsqueeze(0), num_boundary).squeeze(0).long()   #n,
    sampled_boundary_indices = boundary_indices[sampled_boundary_indices].to(device)  # n,

    center_indices = (is_edge == 0).nonzero(as_tuple=True)[1]

    num_center = int(len(center_indices) * 0.1)
    random_indices = random_sample(center_indices, num_center).long()
    sampled_center_coords = coords[0, random_indices]
    final_center_indices = furthest_point_sample(sampled_center_coords.unsqueeze(0), int(num_queries * 0.5)).squeeze(0).long()
    final_center_indices = center_indices[final_center_indices].to(device)
    final_indices = torch.cat((sampled_boundary_indices, final_center_indices))

    return final_indices


