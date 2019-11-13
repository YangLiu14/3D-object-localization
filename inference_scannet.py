import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.models.detection.mask_rcnn
import torchvision.transforms as T

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

SCANNET_INSTANCE_CATEGORY_NAMES = [
    # 0      , # 1  ,  # 2   , # 3    , # 4    , # 5   , # 6
    'cabinet', 'bed', 'chair', 'couch', 'table', 'door', 'window',
    # 7  ,        # 8     ,  # 9             . # 10  , # 11     , # 12
    'bookshelf', 'picture', 'kitchen counter', 'desk', 'curtain', 'N/A',
    # 13            ,  # 14   , # 15  , # 16     , # 17
    'shower curtain', 'toilet', 'sink', 'bathtub', 'mirror'
]

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=num_classes)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def get_prediction(img_path, threshold):
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    pred_class = [SCANNET_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class


def random_colour_masks(image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def instance_segmentation_api(img_path, threshold=0.5, rect_th=2, text_size=0.4, text_th=1):
    masks, boxes, pred_cls = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        text_pos = (np.float32(boxes[i][0][0]), np.float32(boxes[i][1][1]-5))
        cv2.putText(img, pred_cls[i], text_pos, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
        plt.figure(figsize=(20, 30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()

if __name__ == "__main__":
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    # model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'trained_model', 'model_12.pth')))
    model = get_model_instance_segmentation(num_classes=18)
    checkpoint = torch.load(os.path.join(ROOT_DIR, 'trained_model', 'model_12.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    model.eval()

    img_path = os.path.join(ROOT_DIR, 'data/maskrcnn_training/raw_rgb/scene0000_01_3720.jpg')
    instance_segmentation_api(img_path)

