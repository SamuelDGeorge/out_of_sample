import os
import shutil
import tarfile

import numpy as np
import tensorflow as tf

from reader import cifar10


RAW_FILES = [
    "train.bin",
    "test.bin"
]

HEIGHT = 32
WIDTH  = 32
DEPTH  = 3

BYTES_PER = 3074

CLASS_NUM = 100

# Ordered list of subclasses
SUBCLASS_NAMES = [
    "aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
    "household electrical devices", "household furniture", "insects", 
    "large carnivores", "large man-made outdoor things", 
    "large natural outdoor scenes", "large omnivores and herbivores", 
    "medium-sized mammals", "non-insect invertebrates", "people", "reptiles", 
    "small mammals", "trees", "vehicles 1", "vehicles 2"
]

# Ordered list of classes
CLASS_NAMES = [
    "beaver", "dolphin", "otter", "seal", "whale",
    "aquarium fish", "flatfish", "ray", "shark", "trout",
    "orchids", "poppies", "roses", "sunflowers", "tulips",
    "bottles", "bowls", "cans", "cups", "plates",
    "apples", "mushrooms", "oranges", "pears", "sweet peppers",
    "clock", "computer keyboard", "lamp", "telephone", "television",
    "bed", "chair", "couch", "table", "wardrobe",
    "bee", "beetle", "butterfly", "caterpillar", "cockroach",
    "bear", "leopard", "lion", "tiger", "wolf",
    "bridge", "castle", "house", "road", "skyscraper",
    "cloud", "forest", "mountain", "plain", "sea",
    "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
    "fox", "porcupine", "possum", "raccoon", "skunk",
    "crab", "lobster", "snail", "spider", "worm",
    "baby", "boy", "girl", "man", "woman",
    "crocodile", "dinosaur", "lizard", "snake", "turtle",
    "hamster", "mouse", "rabbit", "shrew", "squirrel",
    "maple", "oak", "palm", "pine", "willow",
    "bicycle", "bus", "motorcycle", "pickup truck", "train",
    "lawn-mower", "rocket", "streetcar", "tank", "tractor"
]
# TODO:
# Duplicate the subclass_names
# Sort the class_names
# Apply the same sort to the subclass_names

def main(raw_dir='./data/raw/cifar100/', 
        tfr_dir='./data/tfr/cifar100', 
        validate=True,
        validation_percent=0.2):
    # Since this is virtually identical to cifar-10, but with different classes,
    # We'll just steal the main from cifar-10 and make the appropriate updates

    c100 = cifar10.main(raw_dir=raw_dir, tfr_dir=tfr_dir, validate=validate)

    c100.name = "CIFAR-100"
    c100.websites = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"

    c100.bytes_per_ex  = BYTES_PER
    c100.raw_class_num = CLASS_NUM

    c100.raw_subclass_names = SUBCLASS_NAMES
    c100.raw_class_names = CLASS_NAMES

    c100.training_files = RAW_FILES[0]
    c100.validation_files = []
    c100.testing_files  = RAW_FILES[-1]

    c100.raw_reader = lambda f: reader(f, c100.bytes_per_ex, dtype='uint8')
    c100.extractor  = extract

    return c100


def extract(filepath, dest_dir):
    tarfile.open(filepath, 'r:gz').extractall(dest_dir)

    tar_dir = os.path.join(dest_dir, 'cifar-100-binary')


    for file in RAW_FILES:
        os.rename(
            os.path.join(tar_dir, file),
            os.path.join(dest_dir, file))

    # Clean up
    shutil.rmtree(tar_dir)
    os.remove(filepath)


def reader(filepath, bytes_per, dtype='uint8'):
    data_raw = np.fromfile(filepath, dtype=dtype)

    data_raw = np.reshape(data_raw, (-1,bytes_per))
    _ = data_raw[:,0]
    lbls = data_raw[:,1]
    data = data_raw[:,2:]

    return data, lbls