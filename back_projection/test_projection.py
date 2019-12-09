
import argparse
import os, sys, inspect, time
import random
import torch
import torchnet as tnt
import numpy as np
import itertools
import cv2

import util
import data_util
import torchvision.transforms as transforms
from model import Model2d3d
from enet import create_enet_for_3d
from projection import ProjectionHelper

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = BASE_DIR

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
parser.add_argument('--num_classes', default=42, help='#classes')
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
print(opt)

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES']=str(opt.gpu)

# create camera intrinsics
input_image_dims = [320, 240]
proj_image_dims = [17, 13]
intrinsic = util.make_intrinsic(opt.fx, opt.fy, opt.mx, opt.my)
intrinsic = util.adjust_intrinsic(intrinsic, [opt.intrinsic_image_width, opt.intrinsic_image_height], proj_image_dims)
intrinsic = intrinsic.cuda()
grid_dims = [opt.grid_dimX, opt.grid_dimY, opt.grid_dimZ]
column_height = opt.grid_dimZ
batch_size = opt.batch_size
num_images = opt.num_nearest_images
grid_centerX = opt.grid_dimX // 2
grid_centerY = opt.grid_dimY // 2
color_mean = ENET_TYPES[opt.model2d_type][1]
color_std = ENET_TYPES[opt.model2d_type][2]

# create model
num_classes = opt.num_classes
# model2d_fixed, model2d_trainable, model2d_classifier = create_enet_for_3d(ENET_TYPES[opt.model2d_type], opt.model2d_path, num_classes)
# model = Model2d3d(num_classes, num_images, intrinsic, proj_image_dims, grid_dims, opt.depth_min, opt.depth_max, opt.voxel_size)
projection = ProjectionHelper(intrinsic, opt.depth_min, opt.depth_max, proj_image_dims, grid_dims, opt.voxel_size)
# create loss
criterion_weights = torch.ones(num_classes) 
if opt.class_weight_file:
    criterion_weights = util.read_class_weights_from_file(opt.class_weight_file, num_classes, True)
for c in range(num_classes):
    if criterion_weights[c] > 0:
        criterion_weights[c] = 1 / np.log(1.2 + criterion_weights[c])
print(criterion_weights.numpy())
#raw_input('')
criterion = torch.nn.CrossEntropyLoss(criterion_weights).cuda()
criterion2d = torch.nn.CrossEntropyLoss(criterion_weights).cuda()

_SPLITTER = ','
confusion = tnt.meter.ConfusionMeter(num_classes)
confusion2d = tnt.meter.ConfusionMeter(num_classes)
confusion_val = tnt.meter.ConfusionMeter(num_classes)
confusion2d_val = tnt.meter.ConfusionMeter(num_classes)


from data_util import load_depth_label_pose


def load_frames_multi(data_path, frame_indices, depth_images, color_images, poses, color_mean, color_std):
    # construct files
    # num_images = frame_indices.shape[1] - 2
    num_images = 1
    # scan_names = ['scene' + str(scene_id).zfill(4) + '_' + str(scan_id).zfill(2) for scene_id, scan_id in frame_indices[:,:2].numpy()]
    # scan_names = np.repeat(scan_names, num_images)
    # frame_ids = frame_indices[:, 2:].contiguous().view(-1).numpy()

    depth_files = [os.path.join(data_path, 'depth420.png')]
    color_files = [os.path.join(data_path, 'color420.jpg')]
    pose_files = [os.path.join(data_path,  'pose420.txt')]

    # batch_size = frame_indices.size(0) * num_images
    batch_size = 1
    depth_image_dims = [depth_images.shape[2], depth_images.shape[1]]
    color_image_dims = [color_images.shape[3], color_images.shape[2]]
    normalize = transforms.Normalize(mean=color_mean, std=color_std)
    # load data
    for k in range(batch_size):
        depth_image, color_image, pose = load_depth_label_pose(depth_files[k], color_files[k], pose_files[k], depth_image_dims, color_image_dims, normalize)
        color_images[k] = color_image
        depth_images[k] = torch.from_numpy(depth_image)
        poses[k] = pose


def test_projection():

    mask = torch.cuda.LongTensor(1 * column_height)
    depth_images = torch.cuda.FloatTensor(1, proj_image_dims[1], proj_image_dims[0])
    color_images = torch.cuda.FloatTensor(1, 3, input_image_dims[1], input_image_dims[0])
    camera_poses = torch.cuda.FloatTensor(1, 4, 4)
    label_images = torch.cuda.LongTensor(1, proj_image_dims[1], proj_image_dims[0])

    # load_images
    data_path = os.path.join(ROOT_DIR, 'test_data')
    load_frames_multi(data_path, np.array([420]), depth_images, color_images, camera_poses, color_mean, color_std)


    world_to_grids = np.array(np.loadtxt(ROOT_DIR + '/test_data/world2grid420.txt')).reshape(1, 4, 4)
    world_to_grids = torch.from_numpy(world_to_grids)
    world_to_grids = world_to_grids.type(torch.float32)
    # transforms = world_to_grids[v].unsqueeze(1)
    transforms = world_to_grids.unsqueeze(1)
    transforms = transforms.expand(1, 1, 4, 4).contiguous().view(-1, 4, 4).cuda()
    # compute projection mapping
    print(transforms)
    proj_mapping = [projection.compute_projection(d, c, t) for d, c, t in zip(depth_images, camera_poses, transforms)]

    if None in proj_mapping:  # invalid sample
        # print '(invalid sample)'
        Exception("Did not found proj_mapping")
    proj_mapping = list(zip(*proj_mapping))
    proj_ind_3d = torch.stack(proj_mapping[0])
    proj_ind_2d = torch.stack(proj_mapping[1])

    print("finished")

def main():
    test_projection()


if __name__ == '__main__':
    main()


