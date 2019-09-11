#kitti_data_split.py
import sys

import math
import os
from PIL import Image
import tensorflow as tf
from random import shuffle
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('IMAGE_FOLDER',
                           'kitti_seg/data_semantics/training/image_2',
                           'Folder containing images.')

tf.app.flags.DEFINE_string('LIST_FOLDER',
                           'kitti_seg/data_semantics/training/Segmentation_list',
                           'File containing file names of Image.')

cwd = os.getcwd()
train_file = os.path.join(FLAGS.LIST_FOLDER,'train.txt')
val_file   = os.path.join(FLAGS.LIST_FOLDER,'val.txt')

def main(unused_argv):

    images = [f for f in os.listdir(FLAGS.IMAGE_FOLDER) if os.path.splitext(f)[-1] == '.png'] #reading confined to png files
    images = [i.split('.png', 1)[0] for i in images] #chopping .png files
    shuffle(images) #shuffle to create a randomized datasets
    ratio = int(math.ceil(3*len(images)/4))
    train_images = images[:ratio]
    val_images = images[ratio:]
    #create a file with filenames
    
    
    try:
        if os.path.isfile(train_file) is False:
            f = open(train_file,'w+')
        if os.path.isfile(val_file) is False:
            f = open(val_file,'w+')
        print("New List Files being created '.../train.txt' and '.../val.txt'")
    except:
        print("File already existed... overwriting the buffer")


    with open(train_file, 'w') as f:
        for image in train_images:
            f.write(str(image) + '\n')

    with open(val_file, 'w') as f:
        for image in val_images:
            f.write(str(image) + '\n')                

if __name__ == '__main__': 
    main(sys.argv)