from pycocotools.coco import COCO
import numpy as np
from PIL import Image as PI
import os

def get_coco_file(annotation_file):
    return COCO(annotation_file)

def split_image_category(coco_class, category_string, image_directory, train_directory, validation_directory, train_ratio = 0.75, max_square_size = 331):
    #Set where we are putting images
    cat_train_directory = '{}/{}'.format(train_directory,category_string)
    os.mkdir(cat_train_directory)
    cat_test_directory = '{}/{}'.format(validation_directory,category_string)
    os.mkdir(cat_test_directory)
    
    #Get index of category
    cat_index_ids = coco_class.getCatIds(catNms=[category_string])
    cat_image_ids = coco_class.getImgIds(catIds=cat_index_ids)

    training_length = np.int(train_ratio * len(cat_image_ids))

    for i in range(0,len(cat_image_ids)):
        cat_image = coco_class.loadImgs(cat_image_ids[i])[0]
        I = PI.open('%s/%s'%(image_directory, cat_image['file_name']))
        maxsize = (max_square_size,max_square_size)
        I.thumbnail(maxsize,resample=PI.ANTIALIAS)
        path = '{}/{}'.format(cat_test_directory,cat_image['file_name'])
        if i < training_length:
            path = '{}/{}'.format(cat_train_directory,cat_image['file_name'])
        I.save(path)
