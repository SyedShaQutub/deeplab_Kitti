# deeplab_Kitti

Deeplabv3+ (Rethinking Atrous Convolution for Semantic Image Segmentation) implementation finetuned for Kitti Dataset
Model works on the Cityscape pretrained weights. 

Kitti dataset has 34 classes with background class included. Corresponding logits have been changed to suit the working dataset.

run sh local_test_kitti.sh to download the backbone xception model and train with kitti dataset.
