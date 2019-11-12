# deeplab_Kitti

Deeplabv3+ implementation finetuned for Kitti Dataset
Model works on the Cityscape pretrained weights. 

Kitti dataset has 34 classes with background classes included. Corresponding logits have been changed to suit the working dataset.

run sh local_test_kitti.sh to download the backbone xception model and train with kitti dataset.+++


# Deeplabv3+

**Currently, the project runs on python 2.7 and TensorFlow <=1.8.** 

**Create a new conda env for deeplabv3+**
- Install Conda - https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart
- To create : conda create --name MY_ENV python=2.7
- To remove : conda remove --name Myenv--all

TODO: Migrate to python 3.X and Tensorflow >=2.X

### Required Conda packages:
- **Tensorflow-gpu 1.8** - conda install -c anaconda tensorflow-gpu==1.8
- **Tensorboard**        - conda install -c conda-forge tensorboard
- **Numpy**              - conda install -c anaconda numpy
- **Tqdm** - conda install -c conda-forge tqdm
- **Scipy** - conda install -c anaconda scipy
- **Cudatoolkit** - conda install -c anaconda cudatoolkit=X.X (ex: 10.1)
- **Pillow** - conda install -c anaconda pillow
- **Jupyter** - conda install -c anaconda jupyter

**Note: While running the model, if you feel I have missed some packages to include in the list above, please feel free to inform me.**

### To train a model:

Configure the file local_test.sh file depending on the required parameters.

**example for Kitti dataset** 

- pre-trained model depending on the working dataset and variant of the model.
- crop size for training and evaluation.
- atrous rates, batch size and epochs.



### Models for inference

- The trained and working models are located in /Model_results folder.
- **Demo** - Run deeplab-baseVar.ipynb to test pre-trained model 
- **inference_deeplab_script.py** - Performs semantic segmentation on multiple images at once. If you intend to use this, change the data path accordingly.
- Please add your trained model if you feel the model is useful for the near future for inference or for further training. Create a **Model_folder** with the model name and store the parameters used to train this particular model. Place the export folder [just consisting of the trained model "*.pb"] into the **Model_folder**.




### Issues faced
- Note: if sudo is used to run local_test_*.sh, local python env is loaded instead of conda env. Therefore ignore using Sudo while running this file.




[SemanticSeg_Report.docx](uploads/3649e108f74bbfebf569fe46fa5c4387/SemanticSeg_Report.docx)
