import os
import random
import sys
import numpy as np
import torch
import pickle
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

object_dict = dict()


# helper functions for sanity check
def random_colour_masks(image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def instance_segmentation_api(img, target, threshold=0.5, rect_th=2, text_size=0.4, text_th=1):
    masks, boxes, pred_cls = target['masks'].numpy(), target['boxes'].numpy(), target['labels'].numpy()
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), color=(0, 255, 0), thickness=rect_th)
        text_pos = (np.float32(boxes[i][0]), np.float32(boxes[i][1]-5))
        cv2.putText(img, str(pred_cls[i]), text_pos, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
        plt.figure(figsize=(20, 30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()


class ScannetDataset(Dataset):

    def __init__(self, root, transforms=None, data_split='all'):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, data_split, "raw_rgb"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, data_split, "label_mask"))))
        self.bboxs = list(sorted(os.listdir(os.path.join(root, data_split, "bbox"))))
        self.data_split = data_split

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.data_split, "raw_rgb", self.imgs[idx])
        mask_path = os.path.join(self.root, self.data_split, "label_mask", self.masks[idx])
        bbox_path = os.path.join(self.root, self.data_split, "bbox", self.bboxs[idx])

        img = Image.open(img_path).convert("RGB")
        img = np.asarray(img).astype('float32') / 255.0  # normalize every pixel to 0~1
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # obj_ids = np.unique(mask)
        # # first id is the background, so remove it
        # obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        # masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        # num_objs = len(obj_ids)
        boxes = []
        bbox_dict_list = pickle.load(open(bbox_path, "rb"))

        obj_ids = []
        sem_labels = []
        for bbox_dict in bbox_dict_list:
            # Check: valid bounding-boxes should not have `xmin==xmax or ymin==ymax`
            bbox = bbox_dict['bbox']
            if not (bbox[0] == bbox[2] or bbox[1] == bbox[3]):
                boxes.append(bbox_dict['bbox'])
                sem_labels.append(bbox_dict['sem_label'])
                obj_ids.append(bbox_dict['object_id'] + 1)

            object_dict[bbox_dict['sem_label']] = bbox_dict['object_name']

        if boxes == []:
            Exception("Incomplete data: boxes list is empty!!")

        num_objs = len(obj_ids)
        obj_ids = np.array(obj_ids)
        masks = mask == obj_ids[:, None, None]

        # boxes_test = []
        # for i in range(num_objs):
        #     pos = np.where(masks[i])
        #     xmin = np.min(pos[1])
        #     xmax = np.max(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.max(pos[0])
        #     boxes_test.append([xmin, ymin, xmax, ymax])
        # boxes_test = torch.as_tensor(boxes_test, dtype=torch.float32)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(sem_labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        except:
            Exception("area cannot be calculated.")
        # suppose all instances are not crowd
        # instances with `iscrowd=True` will be ignored during evaluation.
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    print("BASE_DIR: {}".format(BASE_DIR))
    print("ROOT_DIRL {}".format(ROOT_DIR))

    data_path = os.path.join(ROOT_DIR, 'data/maskrcnn_training')
    print("data_path: {}".format(data_path))

    # check how many classes are there
    # check if masks, boxes and classes are correct
    classes = list()
    test_dataset = ScannetDataset(data_path, data_split='train')
    # for i in range(test_dataset.__len__()):
    for i in range(1):
        img, target = test_dataset.__getitem__(i)
        labels = target['labels'].numpy()
        masks = target['masks'].numpy()
        # # store every mask as image
        # i = 0
        # for mask in masks:
        #     im = Image.fromarray(mask*80)
        #     im.save("mask_sanity_check/mask{}.jpeg".format(i))
        #     i += 1
        for label in labels:
            classes.append(label)

        img = img * 255
        img = img.astype('uint8')
        instance_segmentation_api(img, target)

    print(classes)
    print(object_dict)
