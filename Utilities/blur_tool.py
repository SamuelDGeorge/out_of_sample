import random
import decimal
import os
from PIL import Image as PI
from PIL import ImageEnhance
from PIL import ImageFilter
from os.path import isfile
import argparse

def blur_and_print_image(image_full,image_name, output_path):
    file, ext = os.path.splitext(image_full)
    abrev_name, ext = os.path.splitext(image_name)
    image = PI.open(image_full)
    blur_ammount = random.randint(20,60)
    image_blurred = image.filter(ImageFilter.BoxBlur(blur_ammount))
    image_blurred_final = image_blurred.filter(ImageFilter.DETAIL)
    print('File Made: ' + output_path + '/' + abrev_name + "_blurred.jpeg")
    image_blurred_final.save(output_path + '/' + abrev_name + "_blurred.jpeg", "JPEG")


def generate_multiple_images_directory(folder_path, output_path):
    files = os.listdir(folder_path)
    for i in files:
        file_full = folder_path + "/" + i
        file_abreviated = i
        if (isfile(file_full) and file_full.endswith(".JPEG")):
            blur_and_print_image(file_full, file_abreviated, output_path)

