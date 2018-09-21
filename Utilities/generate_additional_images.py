#Manipulate image
import random
import decimal
import os
from PIL import Image as PI
from PIL import ImageEnhance
from os.path import isfile
import argparse

parser = argparse.ArgumentParser(description='Use in a folder to generate multiple images \
            from a single image to train a neural network. Produces 4 additional images \
            with adjustments to orientation, colorscale, and brightness.')
parser.add_argument('--folder', help="Specify the folder in which to manipulate the images.", default=os.getcwd(),
                    type=str, action='store')

def random_brightness(image):
    enhance = ImageEnhance.Brightness(image)
    image_random = enhance.enhance(decimal.Decimal(random.randrange(3,20))/10)
    return image_random

def generate_multiple_images(filename):
    file, ext = os.path.splitext(filename)

    image = PI.open(filename)
    image_90 = image.rotate(90)
    image_180 = image.rotate(180)
    image_270 = image.rotate(270)
    image_grey = image.convert(mode="L")
    image_90_grey = image_90.convert(mode="L")
    image_180_grey = image_180.convert(mode="L")
    image_270_grey = image_270.convert(mode="L")


    image_bright_random = random_brightness(image)
    image_90_bright_random = random_brightness(image_90)
    image_180_bright_random = random_brightness(image_180)
    image_270_bright_random = random_brightness(image_270)

    i = random.randrange(1,3)

    if i == 1:
        image_180.save(file + "_180.jpg", "JPEG")
        image_grey.save(file + "_grey.jpg", "JPEG")
        image_90_bright_random.save(file + "_90_bright.jpg", "JPEG")
        image_180_bright_random.save(file + "_180_bright.jpg", "JPEG")
    elif i == 2:
        image_90.save(file + "_90.jpg", "JPEG")
        image_270_grey.save(file + "_270_grey.jpg", "JPEG")
        image_180_bright_random.save(file + "_180_bright.jpg", "JPEG")
        image_180_grey.save(file + "_180_grey.jpg", "JPEG")
    else:
        image_90_grey.save(file + "_90_grey.jpg", "JPEG")
        image_bright_random.save(file + "_bright.jpg", "JPEG")
        image_90_bright_random.save(file + "_90_bright.jpg", "JPEG")
        image_270_bright_random.save(file + "_270_bright.jpg", "JPEG")
    
def generate_multiple_images_directory(folder_path):
    files = os.listdir(folder_path)
    for i in files:
        file = folder_path + "/" + i
        if (isfile(file) and file.endswith(".jpg")):
            generate_multiple_images(file)

def main():
    args = parser.parse_args()
    if (not os.path.isdir(args.folder)):
        print("Please enter a valid folder to run program!")
        exit(1)
    generate_multiple_images_directory(args.folder)
    print("Generated images on all jpg files in " + args.folder)

if __name__=="__main__":
    main()