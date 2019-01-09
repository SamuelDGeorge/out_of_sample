import random
import decimal
import os
from PIL import Image as PI
from PIL import ImageEnhance
from PIL import ImageFilter
from os.path import isfile
import argparse

def resize_and_print_image(image_full,image_name, output_path, height, width):
    file, ext = os.path.splitext(image_full)
    abrev_name, ext = os.path.splitext(image_name)
    image = PI.open(image_full)
    maxsize = (height,width)
    image.thumbnail(maxsize,resample=PI.ANTIALIAS)
    print('File Made: ' + output_path + '/' + abrev_name + ".jpeg")
    image.save(output_path + '/' + abrev_name + ".jpeg", "JPEG")


def resize_images_in_directory(folder_path, output_path, height, width):
    files = os.listdir(folder_path)
    for i in files:
        file_full = folder_path + "/" + i
        file_abreviated = i
        if (isfile(file_full) and file_full.endswith(".JPEG")):
            resize_and_print_image(file_full, file_abreviated, output_path, height, width)

