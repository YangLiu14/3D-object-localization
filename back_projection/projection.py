
import numpy as np
import torch
from torch.autograd import Function

class ProjectionHelper():
    def __init__(self, intrinsic, depth_min, depth_max, image_dims):
        self.intrinsic = intrinsic
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.image_dims = image_dims


    def depth_to_skeleton(self, ux, uy, depth):
        x = (ux - self.intrinsic[0][2]) / self.intrinsic[0][0]
        y = (uy - self.intrinsic[1][2]) / self.intrinsic[1][1]
        return torch.Tensor([depth*x, depth*y, depth])


    def skeleton_to_depth(self, p):
        x = (p[0] * self.intrinsic[0][0]) / p[2] + self.intrinsic[0][2]
        y = (p[1] * self.intrinsic[1][1]) / p[2] + self.intrinsic[1][2]
        return torch.Tensor([x, y, p[2]])

    def compute_frustum_bounds(self, camera_to_world, axis_align_matrix=None):
        corner_points = camera_to_world.new(8, 4, 1).fill_(1)
        # depth min
        corner_points[0][:3] = self.depth_to_skeleton(0, 0, self.depth_min).unsqueeze(1)
        corner_points[1][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_min).unsqueeze(1)
        corner_points[2][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1,
                                                      self.depth_min).unsqueeze(1)
        corner_points[3][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_min).unsqueeze(1)
        # depth max
        corner_points[4][:3] = self.depth_to_skeleton(0, 0, self.depth_max).unsqueeze(1)
        corner_points[5][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_max).unsqueeze(1)
        corner_points[6][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1,
                                                      self.depth_max).unsqueeze(1)
        corner_points[7][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_max).unsqueeze(1)

        p = torch.bmm(camera_to_world.repeat(8, 1, 1), corner_points)
        # pl = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.floor(p)))
        # pu = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.ceil(p)))

        p = p.squeeze()
        p = p.cpu().numpy()

        if axis_align_matrix is not None:
            pts = np.ones((p.shape[0], 4))
            pts[:, 0:3] = p[:, 0:3]
            pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
            p[:, 0:3] = pts[:, 0:3]

        p = torch.from_numpy(p)

        p = torch.unsqueeze(p, 2)

        bbox_min0, _ = torch.min(p[:, :3, 0], 0)
        # bbox_min1, _ = torch.min(pu[:, :3, 0], 0)
        # bbox_min = np.minimum(bbox_min0, bbox_min1)
        bbox_max0, _ = torch.max(p[:, :3, 0], 0)
        # bbox_max1, _ = torch.max(pu[:, :3, 0], 0)
        # bbox_max = np.maximum(bbox_max0, bbox_max1)
        return bbox_min0, bbox_max0

        # TODO make runnable on cpu as well...
    def compute_projection(self, depth, camera_to_world, axis_align_matrix=None):
        # compute projection by voxels -> image
        world_to_camera = torch.inverse(camera_to_world)
        voxel_bounds_min, voxel_bounds_max = self.compute_frustum_bounds(camera_to_world, axis_align_matrix)
        return voxel_bounds_min, voxel_bounds_max, world_to_camera


        # voxel_bounds_min = np.maximum(voxel_bounds_min, 0).cuda()
        # voxel_bounds_max = np.minimum(voxel_bounds_max, self.volume_dims).float().cuda()
        #
        # # coordinates within frustum bounds
        # lin_ind_volume = torch.arange(0, self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2], out=torch.LongTensor()).cuda()
        # coords = camera_to_world.new(4, lin_ind_volume.size(0))
        # coords[2] = lin_ind_volume / (self.volume_dims[0]*self.volume_dims[1])
        # tmp = lin_ind_volume - (coords[2]*self.volume_dims[0]*self.volume_dims[1]).long()
        # coords[1] = tmp / self.volume_dims[0]
        # coords[0] = torch.remainder(tmp, self.volume_dims[0])
        # coords[3].fill_(1)
        # mask_frustum_bounds = torch.ge(coords[0], voxel_bounds_min[0]) * torch.ge(coords[1], voxel_bounds_min[1]) * torch.ge(coords[2], voxel_bounds_min[2])
        # mask_frustum_bounds = mask_frustum_bounds * torch.lt(coords[0], voxel_bounds_max[0]) * torch.lt(coords[1], voxel_bounds_max[1]) * torch.lt(coords[2], voxel_bounds_max[2])
        # if not mask_frustum_bounds.any():
        #     #print('error: nothing in frustum bounds')
        #     return None
        # lin_ind_volume = lin_ind_volume[mask_frustum_bounds]
        # coords = coords.resize_(4, lin_ind_volume.size(0))
        # coords[2] = lin_ind_volume / (self.volume_dims[0]*self.volume_dims[1])
        # tmp = lin_ind_volume - (coords[2]*self.volume_dims[0]*self.volume_dims[1]).long()
        # coords[1] = tmp / self.volume_dims[0]
        # coords[0] = torch.remainder(tmp, self.volume_dims[0])
        # coords[3].fill_(1)
        #
        # # transform to current frame
        # p = torch.mm(world_to_camera, torch.mm(grid_to_world, coords))
        #
        # # project into image
        # p[0] = (p[0] * self.intrinsic[0][0]) / p[2] + self.intrinsic[0][2]
        # p[1] = (p[1] * self.intrinsic[1][1]) / p[2] + self.intrinsic[1][2]
        # pi = torch.round(p).long()
        #
        # valid_ind_mask = torch.ge(pi[0], 0) * torch.ge(pi[1], 0) * torch.lt(pi[0], self.image_dims[0]) * torch.lt(pi[1], self.image_dims[1])
        # if not valid_ind_mask.any():
        #     #print('error: no valid image indices')
        #     return None
        # valid_image_ind_x = pi[0][valid_ind_mask]
        # valid_image_ind_y = pi[1][valid_ind_mask]
        # valid_image_ind_lin = valid_image_ind_y * self.image_dims[0] + valid_image_ind_x
        # depth_vals = torch.index_select(depth.view(-1), 0, valid_image_ind_lin)
        # depth_mask = depth_vals.ge(self.depth_min) * depth_vals.le(self.depth_max) * torch.abs(depth_vals - p[2][valid_ind_mask]).le(self.voxel_size)
        #
        # if not depth_mask.any():
        #     #print('error: no valid depths')
        #     return None
        #
        # lin_ind_update = lin_ind_volume[valid_ind_mask]
        # lin_ind_update = lin_ind_update[depth_mask]
        # lin_indices_3d = lin_ind_update.new(self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2] + 1) #needs to be same size for all in batch... (first element has size)
        # lin_indices_2d = lin_ind_update.new(self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2] + 1) #needs to be same size for all in batch... (first element has size)
        # lin_indices_3d[0] = lin_ind_update.shape[0]
        # lin_indices_2d[0] = lin_ind_update.shape[0]
        # lin_indices_3d[1:1+lin_indices_3d[0]] = lin_ind_update
        # lin_indices_2d[1:1+lin_indices_2d[0]] = torch.index_select(valid_image_ind_lin, 0, torch.nonzero(depth_mask)[:,0])
        # num_ind = lin_indices_3d[0]
        #
        # return lin_indices_3d, lin_indices_2d

# Inherit from Function
class Projection(Function):

    @staticmethod
    def forward(ctx, label, lin_indices_3d, lin_indices_2d, volume_dims):
        ctx.save_for_backward(lin_indices_3d, lin_indices_2d)
        num_label_ft = 1 if len(label.shape) == 2 else label.shape[0]
        output = label.new(num_label_ft, volume_dims[2], volume_dims[1], volume_dims[0]).fill_(0)
        num_ind = lin_indices_3d[0]
        if num_ind > 0:
            vals = torch.index_select(label.view(num_label_ft, -1), 1, lin_indices_2d[1:1+num_ind])
            output.view(num_label_ft, -1)[:, lin_indices_3d[1:1+num_ind]] = vals
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_label = grad_output.clone()
        num_ft = grad_output.shape[0]
        grad_label.data.resize_(num_ft, 32, 41)
        lin_indices_3d, lin_indices_2d = ctx.saved_variables
        num_ind = lin_indices_3d.data[0]
        vals = torch.index_select(grad_output.data.contiguous().view(num_ft, -1), 1, lin_indices_3d.data[1:1+num_ind])
        grad_label.data.view(num_ft, -1)[:, lin_indices_2d.data[1:1+num_ind]] = vals
        return grad_label, None, None, None

