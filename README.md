# mask-rcnn-for-indoor-objects
Train Mask-RCNN for indoor objects detection

## Dataset
The model in this repository is from torchvision which contains pretrained ResNet50-Backbone and then fine-tuned on [ScanNet](http://kaldir.vc.in.tum.de/scannet_benchmark/) dataset. In order to obtain the data, please follow this [README](https://github.com/facebookresearch/votenet/blob/master/scannet/README.md).

### Data preparation
After downloading the dataset from ScanNet, place them under the `/data` folder. You will have 3 folders:

- **frames_square** for rgb images, depth images and semantics segmentation labels(which is not needed here)
- **scannet_frame_labels** contains pixelwise instance segmentation labels
- **scannet_frame_bbox** contains labels for bounding box, object_id, semantic_label_id

So your `/data` folder will look like this:
- data
  - frames_square
    - scene0000_01
      - color
      - depth
  - scannet_frame_labels
    - scene0000_01
      - instance-filt
        - 0.png
        - 1.png
  - scannet_frame_bbox
    - scene0000_01
      - 0.p
      - 20.p

Run batch_load_scannet_data.pyto organize the data so that it will be easy to use our dataloader to train the model.
