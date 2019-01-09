import os
import shutil
import tarfile

import numpy as np
import tensorflow as tf

from reader import splitter as spltr


RAW_FILES = [
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
    "test_batch.bin"
]

HEIGHT = 32
WIDTH  = 32
DEPTH  = 3

BYTES_PER = 3073

CLASS_NUM = 10

SUBCLASS_NAMES = None

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

def main(raw_dir='./data/raw/cifar10/', 
        tfr_dir='./data/tfr/cifar10', 
        validate=True):
    c10 = spltr.ClassSplitter()

    c10.name = "CIFAR-10"
    c10.websites = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

    # cifar-100: https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz

    c10.height = HEIGHT
    c10.width  = WIDTH
    c10.depth  = DEPTH

    c10.bytes_per_ex  = BYTES_PER
    c10.raw_class_num = CLASS_NUM
    
    c10.raw_subclass_names = SUBCLASS_NAMES
    c10.raw_class_names = CLASS_NAMES

    c10.raw_dir = raw_dir
    c10.testing_files = RAW_FILES[-1]
    if validate:
        c10.training_files   = RAW_FILES[0:4]
        c10.validation_files = RAW_FILES[4]
    else:
        c10.training_files   = RAW_FILES[0:5]
        c10.validation_files = []

    c10.tfr_dir = tfr_dir
    c10.training_tfr   = "training.tfrecords"
    c10.validation_tfr = "validation.tfrecords"
    c10.testing_tfr    = "testing.tfrecords"

    c10.raw_reader = lambda f: spltr.reader(f, c10.bytes_per_ex, dtype='uint8')
    c10.extractor  = extract

    c10.parser = lambda ex: spltr.parser(ex, 
        c10.height, c10.width, c10.depth)
    c10.augmenter = augmenter

    return c10


def augmenter(x):
    if np.random.uniform(size=1) > 0.5:
        data = tf.image.flip_left_right(x[0])
    else:
        data = x[0]

    return data, x[-1]


def extract(filepath, dest_dir):
    tarfile.open(filepath, 'r:gz').extractall(dest_dir)

    tar_dir = os.path.join(dest_dir, 'cifar-10-batches-bin')


    for file in RAW_FILES:
        os.rename(
            os.path.join(tar_dir, file),
            os.path.join(dest_dir, file))

    # Clean up
    shutil.rmtree(tar_dir)
    os.remove(filepath)