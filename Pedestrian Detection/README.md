# Pedestrian Detection Using Mask R-CNN using Pytorch

In this project, I finetuned Mask R-CNN model to detect only pedestrians in a image. I trained this model on Penn Fudan Database. I used MaskR-CNN_Resnet50_fpn model from torchvision.models and tuned it to detect 2 classes- Pedestrians and Background.

__Requirements__
- Pytorch
- Numpy
- Matplotlib

__Dataset__
[Penn-Fudan Database for Pedestrian Detection and Segmentation](https://www.cis.upenn.edu/~jshi/ped_html/)
This dataset contains 170 images with 345 labelled pedestrians(masks and bounding boxes)
To download the dataset [Click here](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip) 