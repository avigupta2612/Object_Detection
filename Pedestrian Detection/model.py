from torchvision import models

def get_model_instance(num_classes):
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    #print(model)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer= 256
    model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask,
                                                                                  hidden_layer,
                                                                                  num_classes)
    
    return model