# Pedestrian Detection Using Mask R-CNN using Pytorch

In this project, I finetuned Mask R-CNN model to detect only pedestrians in a image. I trained this model on Penn Fudan Database. I used MaskR-CNN_Resnet50_fpn model from torchvision.models and tuned it to detect 2 classes- Pedestrians and Background.<br/>
To run this project first download dataset by running dataset/dataset.sh and get helper files by running helper_files.sh. Then run train.py function to train the model.<br/>
Or check out the [jupyter notebook](https://github.com/avigupta2612/Object_Detection/blob/master/Pedestrian%20Detection/Pedestrian_Detection_MaskRCNN.ipynb) for complete tutorial.

__Requirements__
- Pytorch
- Numpy
- Matplotlib

__Dataset__<br/>
[Penn-Fudan Database for Pedestrian Detection and Segmentation](https://www.cis.upenn.edu/~jshi/ped_html/)
This dataset contains 170 images with 345 labelled pedestrians(masks and bounding boxes).<br/>
To download the dataset [Click here](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip) 