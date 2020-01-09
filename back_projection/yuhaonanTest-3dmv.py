import argparse
import os, sys, inspect, time
import random
import torch
import torchnet as tnt
import numpy as np
import itertools
import cv2
import imageio
import torchvision
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from PIL import Image
import torchvision.transforms.functional as TF
import util
import data_util
import pc_util
import torchvision.transforms as transforms

from tqdm import tqdm
from model import Model2d3d
from enet import create_enet_for_3d
from projection import ProjectionHelper
from data_util import load_depth_label_pose
from scannet_utils import read_mesh_vertices

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# train on the GPU or on the CPU, if a GPU is not available
if torch.cuda.is_available():
    print("Using GPU to train")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ENET_TYPES = {'scannet': (41, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}  #classes, color mean/std

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--train_data_list', default='data/', required=False, help='path to file list of h5 train data')
parser.add_argument('--val_data_list', default='', help='path to file list of h5 val data')
parser.add_argument('--output', default='./logs', help='folder to output model checkpoints')
parser.add_argument('--data_path_2d', required=False, help='path to 2d train data')
parser.add_argument('--class_weight_file', default='', help='path to histogram over classes')
# train params
parser.add_argument('--num_classes', default=18, help='#classes')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--max_epoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum, default=0.9')
parser.add_argument('--num_nearest_images', type=int, default=3, help='#images')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay, default=0.0005')
parser.add_argument('--retrain', default='', help='model to load')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--model2d_type', default='scannet', help='which enet (scannet)')
parser.add_argument('--model2d_path', required=False, help='path to enet model')
parser.add_argument('--use_proxy_loss', dest='use_proxy_loss', action='store_true')
# 2d/3d
parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size (in meters)')
parser.add_argument('--grid_dimX', type=int, default=31, help='3d grid dim x')
parser.add_argument('--grid_dimY', type=int, default=31, help='3d grid dim y')
parser.add_argument('--grid_dimZ', type=int, default=62, help='3d grid dim z')
parser.add_argument('--depth_min', type=float, default=0.4, help='min depth (in meters)')
parser.add_argument('--depth_max', type=float, default=4.0, help='max depth (in meters)')
# scannet intrinsic params
parser.add_argument('--intrinsic_image_width', type=int, default=640, help='2d image width')
parser.add_argument('--intrinsic_image_height', type=int, default=480, help='2d image height')
parser.add_argument('--fx', type=float, default=577.870605, help='intrinsics')
parser.add_argument('--fy', type=float, default=577.870605, help='intrinsics')
parser.add_argument('--mx', type=float, default=319.5, help='intrinsics')
parser.add_argument('--my', type=float, default=239.5, help='intrinsics')

parser.set_defaults(use_proxy_loss=False)
opt = parser.parse_args()
assert opt.model2d_type in ENET_TYPES

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

def read_lines_from_file(filename):
    assert os.path.isfile(filename)
    lines = open(filename).read().splitlines()
    return lines

# create camera intrinsics
input_image_dims = [320, 240]
proj_image_dims = [17, 13]
# Read intrinsics from txt file
# TODO: read scene_id on the during the trainning
scene_path = "../data/scenes/"
rawdata_path = "../data/rawdata/"
num_classes = opt.num_classes
color_mean = ENET_TYPES[opt.model2d_type][1]
color_std = ENET_TYPES[opt.model2d_type][2]


def get_features_for_projection(model, imagePath, device):
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    # these transform parameters are from source code of Mask R CNN
    transform = GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=image_mean, image_std=image_std)
    image = Image.open(imagePath)
    image_tensor = TF.to_tensor(image)
    # let it be in list (can be multiple)
    # TODO make it multiple
    images = [image_tensor]
    images, _ = transform(images)
    features = model.backbone(images.tensors.to(device))
    features_to_be_projected = features['pool']
    return features_to_be_projected

def get_features_for_projection_multi(model, imagePaths, device):
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    # these transform parameters are from source code of Mask R CNN
    transform = GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=image_mean, image_std=image_std)
    images = [Image.open(imagePath) for imagePath in imagePaths]
    image_tensors = [TF.to_tensor(image) for image in images]
    # let it be in list (can be multiple)
    results = []
    with torch.no_grad():
        for tensor in image_tensors:
            images, _ = transform([tensor])
            features = model.backbone(images.tensors.to(device))
            features_to_be_projected = features['pool']
            results.append(features_to_be_projected[0])
    return results

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=num_classes)
    return model

def load_frames_multi(data_path, image_name, depth_image, color_image, camera_pose, color_mean, color_std):
    print(image_name)
    depth_file = os.path.join(data_path, 'depth', image_name+".png")
    color_file = os.path.join(data_path, 'color', image_name+".jpg")
    pose_file = os.path.join(data_path,  'pose', image_name+".txt")
    depth_image_dims = [depth_image.shape[2], depth_image.shape[1]]
    color_image_dims = [color_image.shape[3], color_image.shape[2]]
    normalize = transforms.Normalize(mean=color_mean, std=color_std)
    # load data
    depth_img, color_img, pose = load_depth_label_pose(depth_file, color_file, pose_file, depth_image_dims, color_image_dims, normalize)
    color_image = color_img
    depth_image = torch.from_numpy(depth_img)
    camera_pose = pose
    return color_image,depth_image,camera_pose

def load_frames_multi_2(data_path, image_names, depth_image_torch, color_image_torch, camera_pose, color_mean, color_std):
    color_images = []
    depth_images = []
    camera_poses = []
    for image_name in image_names:
        depth_file = os.path.join(data_path, 'depth', image_name+".png")
        color_file = os.path.join(data_path, 'color', image_name+".jpg")
        pose_file = os.path.join(data_path,  'pose', image_name+".txt")
        depth_image_dims = [depth_image_torch.shape[2], depth_image_torch.shape[1]]
        color_image_dims = [color_image_torch.shape[3], color_image_torch.shape[2]]
        normalize = transforms.Normalize(mean=color_mean, std=color_std)
        # load data
        depth_img, color_img, pose = load_depth_label_pose(depth_file, color_file, pose_file, depth_image_dims, color_image_dims, normalize)
        color_image = color_img
        depth_image = torch.from_numpy(depth_img)
        camera_pose = pose
        color_images.append(color_image)
        depth_images.append(depth_image)
        camera_poses.append(camera_pose)
    return color_images,depth_images,camera_poses


def axis_align(mesh_vertices, axis_align_matrix):
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    mesh_vertices[:, 0:3] = pts[:, 0:3]
    return mesh_vertices

def project_images(data_path, image_names, mesh_vertices):
    depth_image = torch.cuda.FloatTensor(len(image_names), proj_image_dims[1], proj_image_dims[0])
    color_image = torch.cuda.FloatTensor(len(image_names), 3, input_image_dims[1], input_image_dims[0])
    camera_pose = torch.cuda.FloatTensor(len(image_names), 4, 4)
    color_images,depth_images,camera_poses = load_frames_multi_2(data_path, image_names, depth_image, color_image, camera_pose, color_mean, color_std)
    color_images = torch.stack(color_images)
    depth_images = torch.stack(depth_images)
    camera_poses = torch.stack(camera_poses)
    boundingmin, boundingmax, world_to_camera = projection.compute_projection_multi(camera_poses)
    lin_ind_volume = torch.arange(0, mesh_vertices.shape[0],out=torch.LongTensor()).cuda()
    N = mesh_vertices.shape[0]
    mesh_vertices_original = mesh_vertices.copy()
    mesh_vertices_original = np.transpose(mesh_vertices_original)
    mesh_vertices_original = torch.from_numpy(mesh_vertices_original).float().to(device).cuda()
    mesh_vertices = mesh_vertices[:, 0:4]
    mesh_vertices = np.transpose(mesh_vertices)
    boundingmax = boundingmax.cuda()
    boundingmin = boundingmin.cuda()
    world_to_camera = world_to_camera.cuda()
    mesh_vertices = torch.from_numpy(mesh_vertices).float().to(device).cuda()
    mesh_vertices = mesh_vertices.repeat(len(image_names),1,1)
    mask_frustum_bounds = torch.ge(mesh_vertices[:,0], boundingmin[:,0].unsqueeze(1)) * torch.ge(mesh_vertices[:,1], boundingmin[:,1].unsqueeze(1)) * torch.ge(mesh_vertices[:,2], boundingmin[:,2].unsqueeze(1))
    mask_frustum_bounds = mask_frustum_bounds * torch.lt(mesh_vertices[:,0], boundingmax[:,0].unsqueeze(1)) * torch.lt(mesh_vertices[:,1], boundingmax[:,1].unsqueeze(1)) * torch.lt(mesh_vertices[:,2], boundingmax[:,2].unsqueeze(1))
    if not mask_frustum_bounds.any():
        # print('error: nothing in frustum bounds')
        return None
    lin_ind_volume = lin_ind_volume.repeat(len(image_names),1)
    # lin_ind_volume = lin_ind_volume[mask_frustum_bounds]
    p = torch.bmm(world_to_camera, mesh_vertices)
    mesh_vertices = mesh_vertices.permute(1,0,2)
    mesh_vertices = mesh_vertices[:,mask_frustum_bounds]
    # transform to current frame
    p = p.permute(1, 0, 2)
    # p = p[:,mask_frustum_bounds]
    # project into image
    p[0] = (p[0] * projection.intrinsic[0][0]) / p[2] + projection.intrinsic[0][2]
    p[1] = (p[1] * projection.intrinsic[1][1]) / p[2] + projection.intrinsic[1][2]
    pi = torch.round(p).long()
    valid_ind_mask = torch.ge(pi[0,:], 0) * torch.ge(pi[1,:], 0) * torch.lt(pi[0,:], proj_image_dims[0]) * torch.lt(pi[1,:], proj_image_dims[1])
    if not valid_ind_mask.any():
        # print('error: no valid image indices')
        return None

    pi_test = pi
    pi_test = pi_test*valid_ind_mask.int().long()
    image_ind_x = pi_test[0,:]
    image_ind_y = pi_test[1,:]
    image_ind_lin = image_ind_y * proj_image_dims[0] + image_ind_x
    depth = depth_images.cuda()
    depth_vals = torch.cat([torch.index_select(a.view(-1), 0, i).unsqueeze(0) for a, i in zip(depth, image_ind_lin)])
    depth_mask = depth_vals.ge(opt.depth_min) * depth_vals.le(opt.depth_max) * torch.abs(
        depth_vals - p[2]*valid_ind_mask.int().float()).le(opt.voxel_size)

    if not depth_mask.any():
        # print('error: no valid depths')
        return None
    final_mask = (valid_ind_mask*mask_frustum_bounds*depth_mask)
    lin_indices_3d = [a[i].unsqueeze(0) for a, i in zip(lin_ind_volume, final_mask)]
    lin_indices_2d = [torch.index_select(a, 0, torch.nonzero(i)[:, 0]).unsqueeze(0) for a, i in zip(image_ind_lin, final_mask)]

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    features_to_add = get_features_for_projection_multi(model, [os.path.join(data_path, 'color', image_name+".jpg") for image_name in image_names], device)
    num_label_ft = 1 if len(features_to_add[0].shape) == 2 else features_to_add[0].shape[0]
    output = features_to_add[0].new(4 + num_label_ft, N).fill_(0)
    output[0:4, :] = mesh_vertices_original[0:4, :]
#    num_ind = lin_indices_3d[0]
#    if num_ind > 0:
    for feature, lin_index_2d, lin_index_3d in zip(features_to_add, lin_indices_2d, lin_indices_3d):
        vals = torch.index_select(feature.view(num_label_ft, -1), 1, lin_index_2d[0])
        output.view(num_label_ft+4, -1)[4:, lin_index_3d[0]] = vals
    output = output.detach().cpu().numpy()
    output = np.transpose(output)
    return output

def project_one_image(data_path, image_name, mesh_vertices):
    depth_image = torch.cuda.FloatTensor(1, proj_image_dims[1], proj_image_dims[0])
    color_image = torch.cuda.FloatTensor(1, 3, input_image_dims[1], input_image_dims[0])
    camera_pose = torch.cuda.FloatTensor(1, 4, 4)
    color_image,depth_image,camera_pose = load_frames_multi(data_path, image_name, depth_image, color_image, camera_pose, color_mean, color_std)

    boundingmin, boundingmax, world_to_camera = projection.compute_projection(camera_pose)
    lin_ind_volume = torch.arange(0, mesh_vertices.shape[0],out=torch.LongTensor()).cuda()
    N = mesh_vertices.shape[0]
    mesh_vertices_original = mesh_vertices.copy()
    mesh_vertices_original = np.transpose(mesh_vertices_original)
    mesh_vertices_original = torch.from_numpy(mesh_vertices_original).float().to(device).cuda()
    mesh_vertices = mesh_vertices[:, 0:4]
    mesh_vertices = np.transpose(mesh_vertices)
    boundingmax = boundingmax.cuda()
    boundingmin = boundingmin.cuda()
    world_to_camera = world_to_camera.cuda()
    mesh_vertices = torch.from_numpy(mesh_vertices).float().to(device).cuda()
    mask_frustum_bounds = torch.ge(mesh_vertices[0], boundingmin[0]) * torch.ge(mesh_vertices[1], boundingmin[1]) * torch.ge(mesh_vertices[2], boundingmin[2])
    mask_frustum_bounds = mask_frustum_bounds * torch.lt(mesh_vertices[0], boundingmax[0]) * torch.lt(mesh_vertices[1], boundingmax[1]) * torch.lt(mesh_vertices[2], boundingmax[2])
    if not mask_frustum_bounds.any():
        # print('error: nothing in frustum bounds')
        return None
    lin_ind_volume = lin_ind_volume[mask_frustum_bounds]
    mesh_vertices = mesh_vertices[:, mask_frustum_bounds]
    # transform to current frame
    p = torch.mm(world_to_camera, mesh_vertices)
    # project into image
    p[0] = (p[0] * projection.intrinsic[0][0]) / p[2] + projection.intrinsic[0][2]
    p[1] = (p[1] * projection.intrinsic[1][1]) / p[2] + projection.intrinsic[1][2]
    pi = torch.round(p).long()
    valid_ind_mask = torch.ge(pi[0], 0) * torch.ge(pi[1], 0) * torch.lt(pi[0], proj_image_dims[0]) * torch.lt(pi[1], proj_image_dims[1])
    if not valid_ind_mask.any():
        # print('error: no valid image indices')
        return None
    valid_image_ind_x = pi[0][valid_ind_mask]
    valid_image_ind_y = pi[1][valid_ind_mask]
    valid_image_ind_lin = valid_image_ind_y * proj_image_dims[0] + valid_image_ind_x
    depth = depth_image.cuda()
    depth_vals = torch.index_select(depth.view(-1), 0, valid_image_ind_lin)
    depth_mask = depth_vals.ge(opt.depth_min) * depth_vals.le(opt.depth_max) * torch.abs(
        depth_vals - p[2][valid_ind_mask]).le(opt.voxel_size)

    if not depth_mask.any():
        # print('error: no valid depths')
        return None
    lin_ind_update = lin_ind_volume[valid_ind_mask]
    lin_ind_update = lin_ind_update[depth_mask]
    lin_indices_3d = lin_ind_update.new(N + 1)  # needs to be same size for all in batch... (first element has size)
    lin_indices_2d = lin_ind_update.new(N + 1)  # needs to be same size for all in batch... (first element has size)
    lin_indices_3d[0] = lin_ind_update.shape[0]
    lin_indices_2d[0] = lin_ind_update.shape[0]
    lin_indices_3d[1:1 + lin_indices_3d[0]] = lin_ind_update
    lin_indices_2d[1:1 + lin_indices_2d[0]] = torch.index_select(valid_image_ind_lin, 0, torch.nonzero(depth_mask)[:, 0])

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    features_to_add = get_features_for_projection(model, os.path.join(data_path, 'color', image_name+".jpg"), device)
    features_to_add = features_to_add[0]
    num_label_ft = 1 if len(features_to_add.shape) == 2 else features_to_add.shape[0]
    output = mesh_vertices_original
    num_ind = lin_indices_3d[0]
    if num_ind > 0:
        vals = torch.index_select(features_to_add.view(num_label_ft, -1), 1, lin_indices_2d[1:1 + num_ind])
        output.view(num_label_ft+4, -1)[4:, lin_indices_3d[1:1 + num_ind]] = vals
    output = output.detach().cpu().numpy()
    output = np.transpose(output)
    return output

















    # # filter out the vertices that are not in the frustum (here treated as box-shape)
    # filter1 = mesh_vertices[:, 0]
    # filter1 = (filter1 >= boundingmin[0]) & (filter1 <= boundingmax[0])
    # filter2 = mesh_vertices[:, 1]
    # filter2 = (filter2 >= boundingmin[1]) & (filter2 <= boundingmax[1])
    # filter3 = mesh_vertices[:, 2]
    # filter3 = (filter3 >= boundingmin[2]) & (filter3 <= boundingmax[2])
    # filter_all = filter1 & filter2 & filter3
    # # valid_vertices = mesh_vertices
    # valid_vertices = mesh_vertices[filter_all]
    # valid_indices = np.where(filter_all == True)[0]
    # # valid_indices = np.arange(mesh_vertices.shape[0])
    # # transform to current frame
    # world_to_camera = world_to_camera.cpu().numpy()
    # first_four_columns_of_points = valid_vertices[:,0:4]
    # N = first_four_columns_of_points.shape[0]
    # first_four_columns_of_points_T = np.transpose(first_four_columns_of_points)
    # pcamera = np.matmul(world_to_camera, first_four_columns_of_points_T)  # (4,4) x (4,N) => (4,N)
    #
    # # train on the GPU or on the CPU, if a GPU is not available
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #
    # # get the model using our helper function
    # model = get_model_instance_segmentation(num_classes)
    # # move model to the right device
    # model.to(device)
    #
    # # construct an optimizer
    # params = [param for param in model.parameters() if param.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.001,
    #                              momentum=0.9, weight_decay=0.0005)
    # # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # features_to_add = get_features_for_projection(model, os.path.join(data_path, 'color', image_name+".jpg"), device)
    # p = np.transpose(pcamera)  # shape: (N,4)
    #
    # # project into image
    # p[:, 0] = (p[:, 0] * projection.intrinsic[0][0].cpu().numpy()) / p[:, 2] + projection.intrinsic[0][2].cpu().numpy()
    # p[:, 1] = (p[:, 1] * projection.intrinsic[1][1].cpu().numpy()) / p[:, 2] + projection.intrinsic[1][2].cpu().numpy()
    # pi = np.rint(p)
    #
    # depth_map_to_compare = depth_image.cpu().numpy()
    #
    # p = np.concatenate((p[:], np.zeros(((p.shape[0], 256)))), axis=1)
    #
    # for i1 in range(17):
    #     for i2 in range(13):
    #         x = pi[:, 0]
    #         x_filter = x == i1
    #         y = pi[:, 1]
    #         y_filter = y == i2
    #         correct_depth = depth_map_to_compare[i2][i1]
    #         depth = p[:, 2]
    #         depth_filter = (depth <= correct_depth+0.05) & (depth >= correct_depth-0.05)
    #
    #         filterxyd = x_filter & y_filter & depth_filter
    #
    #         projected_indices = valid_indices[filterxyd]
    #
    #         p_temp = p[filterxyd]
    #         if p_temp.shape[0] != 0:
    #             features = features_to_add[0,:,i2,i1].detach().cpu().numpy()
    #             N_points_to_project = p_temp.shape[0]
    #             tiled_features = np.tile(features, (N_points_to_project, 1))
    #             featured_point = np.concatenate((p[filterxyd,0:4], tiled_features), axis=1)
    #             mesh_vertices[projected_indices, 4:] = featured_point[:, 4:]

    return mesh_vertices

# ROOT_DIR:indoor-objects
def scannet_projection(intrinsic, projection, scene_id):
    # load vertices
    mesh_vertices = read_mesh_vertices(filename=ROOT_DIR + '/data/scenes/' + scene_id + '/' + scene_id + '_vh_clean_2.ply')
    # add addition ones to the end or each position in mesh_vertices
    mesh_vertices = np.append(mesh_vertices, np.ones((mesh_vertices.shape[0], 1)), axis=1)
    # add zeros dimension
    mesh_vertices = np.concatenate((mesh_vertices[:], np.zeros(((mesh_vertices.shape[0], 256)))), axis=1)
    # load_images
    image_path = os.path.join(ROOT_DIR, 'data', 'rawdata', scene_id, 'color')
    # test multiple ==============================
    images = []
    # for image_name in tqdm(os.listdir(image_path)):
    #     image_name = image_name.replace(".jpg", "", 1)
    #     images.append(image_name)
    # data_path = os.path.join(ROOT_DIR, 'data', 'rawdata', scene_id)
    # mesh_vertices_projected = project_images(data_path, images, mesh_vertices)
    # test==============================


    for image_name in tqdm(os.listdir(image_path)):
        image_name = image_name.replace(".jpg", "", 1)
        data_path = os.path.join(ROOT_DIR, 'data', 'rawdata', scene_id)
        mesh_vertices_projected = project_one_image(data_path, image_name, mesh_vertices)
        # mesh_vertices = np.concatenate((projected_points,unvalid_vertices),axis=0)
        mesh_vertices = mesh_vertices_projected
        # check = mesh_vertices[:, 4]
        # check = check != 0.0
        # check = mesh_vertices[check]


    # Load alignments
    lines = open(ROOT_DIR + '/data/scenes/'+scene_id+'/'+scene_id+ '.txt').readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                                 for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    # delete all 1's column
    mesh_vertices = np.delete(mesh_vertices, 3, 1)
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    mesh_vertices[:, 0:3] = pts[:, 0:3]
    mesh_vertices = mesh_vertices.astype('float32')
    return mesh_vertices













if __name__ == '__main__':
    for scene_id in os.listdir(scene_path):
        save_path = "/home/haonan/Desktop/test-maskrcnn/"
        print(scene_id)
        intrinsic_str = read_lines_from_file(rawdata_path + scene_id + '/intrinsic_depth.txt')
        fx = float(intrinsic_str[0].split()[0])
        fy = float(intrinsic_str[1].split()[1])
        mx = float(intrinsic_str[0].split()[2])
        my = float(intrinsic_str[1].split()[2])
        intrinsic = util.make_intrinsic(fx, fy, mx, my)
        intrinsic = util.adjust_intrinsic(intrinsic, [opt.intrinsic_image_width, opt.intrinsic_image_height],
                                          proj_image_dims)
        intrinsic = intrinsic.cuda()
        projection = ProjectionHelper(intrinsic, opt.depth_min, opt.depth_max, proj_image_dims)
        vertices = scannet_projection(intrinsic, projection, scene_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path+scene_id + "_features.npy", vertices)
    print("finished")
