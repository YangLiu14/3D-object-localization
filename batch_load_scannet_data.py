import os
import datetime
import numpy as np
import shutil
import pickle
import cv2
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

SCANNET_RGB_DIR = os.path.join(ROOT_DIR, 'data/frames_square/')
SCANNET_MASK_DIR = os.path.join(ROOT_DIR, 'data/scannet_frame_labels/')
SCANNET_BOX_DIR = os.path.join(ROOT_DIR, 'data/scannet_frame_bbox/')

OUTPUT_FOLDER = os.path.join(ROOT_DIR, 'data/maskrcnn_training/')
ALL_SCAN_NAMES = [line.rstrip() for line in open('meta_data/scannet_all_scans.txt')]
TRAIN_SCAN_NAMES = [line.rstrip() for line in open('meta_data/scannetv2_train.txt')]
VAL_SCAN_NAMES = [line.rstrip() for line in open('meta_data/scannetv2_val.txt')]
TEST_SCAN_NAMES = [line.rstrip() for line in open('meta_data/scannetv2_test.txt')]

IMG_W = 320
IMG_H = 240

def export_one_scan(scan_name, output_dir):
    rgb_folder = os.path.join(SCANNET_RGB_DIR, scan_name, 'color')
    mask_folder = os.path.join(SCANNET_MASK_DIR, scan_name, 'instance-filt')
    bbox_folder = os.path.join(SCANNET_BOX_DIR, scan_name)

    rgb_fnames = os.listdir(rgb_folder)
    mask_fnames_all = os.listdir(mask_folder)
    bbox_fnames = os.listdir(bbox_folder)

    # check correspondence of filenames of [rgb, mask, bbox], and filter out redundant mask_files
    assert len(rgb_fnames) == len(bbox_fnames)
    mask_fnames = list()
    rgb_remove_list = list()
    mask_remove_list = list()
    bbox_remove_list = list()
    for i in range(len(rgb_fnames)):
        fname = rgb_fnames[i].split('.')[0]

        # remove files that have no bounding box annotation
        bbox_dict_list = pickle.load( open(os.path.join(SCANNET_BOX_DIR, scan_name, fname+'.p'), "rb"))
        for bbox_dict in bbox_dict_list:
            # Check: valid bounding-boxes should not have `xmin==xmax or ymin==ymax`
            bbox = bbox_dict['bbox']
            if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                bbox_dict_list.remove(bbox_dict)
        if bbox_dict_list == []:
            rgb_remove_list.append(fname + '.jpg')
            mask_remove_list.append(fname + '.png')
            bbox_remove_list.append(fname + '.p')


        if fname+'.p' not in bbox_fnames:
            print("{} not exist!".format(SCANNET_BOX_DIR + fname + '.p'))
            rgb_remove_list.append(fname + '.jpg')
            continue
        if fname+'.png' in mask_fnames_all:
            mask_fnames.append(fname+'.png')
        else:
            rgb_remove_list.append(fname + '.jpg')
            bbox_remove_list.append(fname + '.p')

    # remove inconsistent files
    for fname in rgb_remove_list:
        rgb_fnames.remove(fname)
    for fname in mask_remove_list:
        mask_fnames.remove(fname)
    for fname in bbox_remove_list:
        bbox_fnames.remove(fname)

    # ======================================================
    # Store images and mask-labels in npy
    # ======================================================

    # img_npy_fpath = os.path.join(OUTPUT_FOLDER, scan_name + "_rgb.npy")
    # im_array = []
    # mask_npy_fpath = os.path.join(OUTPUT_FOLDER, scan_name + "_mask.npy")
    # mask_array = []
    #
    # for fname in rgb_fnames:
    #     fname = fname.split('.')[0]
    #     # RGB
    #     image = cv2.imread(os.path.join(SCANNET_RGB_DIR, scan_name, 'color', fname+'.jpg'))
    #     im_array.append(image)
    #     # label-mask
    #     mask = cv2.imread(os.path.join(SCANNET_MASK_DIR, scan_name, 'instance-filt', fname+'.png'))
    #     mask = cv2.resize(mask, (IMG_W, IMG_H))
    #     mask_array.append(mask)
    #
    # np.save(img_npy_fpath, np.array(im_array))
    # np.save(mask_npy_fpath, np.array(mask_array))

    # ======================================================
    # ======================================================

    # ======================================================
    # Store images and mask-labels in organized folders
    # ======================================================

    # copy and rename files
    for fname in rgb_fnames:
        dst = os.path.join(output_dir, 'raw_rgb/', '_'.join([scan_name, fname]))
        if not os.path.exists(os.path.join(output_dir, 'raw_rgb/')):
            os.mkdir(os.path.join(output_dir, 'raw_rgb/'))
        # dst = os.path.join(OUTPUT_FOLDER, 'raw_rgb/', scan_name)
        # if not os.path.exists(dst):
        #     os.makedirs(dst)
        shutil.copy(rgb_folder + "/" + fname, dst)

    for fname in mask_fnames:
        dst = os.path.join(output_dir, 'label_mask/', '_'.join([scan_name, fname]))
        if not os.path.exists(os.path.join(output_dir, 'label_mask/')):
            os.mkdir(os.path.join(output_dir, 'label_mask/'))
        # dst = os.path.join(OUTPUT_FOLDER, 'label_mask/', scan_name)
        # if not os.path.exists(dst):
        #     os.makedirs(dst)
        im = Image.open(os.path.join(SCANNET_MASK_DIR, scan_name, 'instance-filt', fname))
        im_resized = im.resize((IMG_W, IMG_H))
        # im_resized.save(dst + '/' + fname)
        im_resized.save(dst)

    for fname in bbox_fnames:
        dst = os.path.join(output_dir, 'bbox/', '_'.join([scan_name, fname]))
        if not os.path.exists(os.path.join(output_dir, 'bbox/')):
            os.mkdir(os.path.join(output_dir, 'bbox/'))
        # bbox = pickle.load( open(os.path.join(SCANNET_BOX_DIR, scan_name, fname), "rb"))
        shutil.copy(bbox_folder + "/" + fname, dst)

    # ======================================================
    # ======================================================


def batch_export(data_split):
    output_dir = ''
    scan_names = ''
    if data_split == 'train':
        output_dir = os.path.join(OUTPUT_FOLDER, 'train')
        scan_names = TRAIN_SCAN_NAMES
    elif data_split == 'valid':
        output_dir = os.path.join(OUTPUT_FOLDER, 'valid')
        scan_names = VAL_SCAN_NAMES
    elif data_split == 'test':
        output_dir = os.path.join(OUTPUT_FOLDER, 'test')
        scan_names = TEST_SCAN_NAMES
    elif data_split == 'all':
        output_dir = os.path.join(OUTPUT_FOLDER, 'all')
        scan_names = ALL_SCAN_NAMES
    else:
        Exception("Invalid `data_split`, please choose from: `all`, `train`,`valid`, `test`")

    if not os.path.exists(output_dir):
        print('Creating new data folder: {}'.format(output_dir))
        os.mkdir(output_dir)

    for scan_name in scan_names:
        print('-' * 20 + 'begin')
        print(datetime.datetime.now())
        print(scan_name)

        # if os.path.isfile(output_filename_prefix + '_vert.npy'):
        #     print('File already exists. skipping.')
        #     print('-' * 20 + 'done')
        #     continue

        # export_one_scan(scan_name, output_filename_prefix)
        try:
            export_one_scan(scan_name, output_dir)
        except:
            print('Failed export scan: %s' % scan_name)
        print('-' * 20 + 'done')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--data-split', default='valid', help='valid options: `all`, `train`,`valid`, `test` ')
    args = parser.parse_args()
    batch_export(args.data_split)

    # # TEST
    # fpath = OUTPUT_FOLDER + '/scene0000_00_rgb.npy'
    # img_array = np.load(fpath)
    # mask_path = OUTPUT_FOLDER + '/scene0000_00_mask.npy'
    # mask_array = np.load(mask_path)
    # assert len(img_array) == len(mask_array)
    # count = 0
    # # TODO: fix PIL.Image read image as BGR
    # for i in range(5):
    #     im = Image.fromarray(img_array[i])
    #     # im.show()
    #     im.save(OUTPUT_FOLDER + '/raw_{}.jpg'.format(count))
    #
    #     mask = Image.fromarray(mask_array[i])
    #     # mask.show()
    #     mask.save(OUTPUT_FOLDER + '/mask_{}.png'.format(count))
    #     count += 1


