import csv
import json
import os
import random
from shutil import copyfile

import cv2
import pandas as pd

from tools.aws.awsTools import Bucket
from tools.image.imageTools import get_img_shape, add_blurring, add_noise, image_read, image_write


def clean_via_file(via_file: str) -> str:
    """
    Cleans annotation csv file from via to differentiate the separators "," from the actual ","
    Returns the path to the clean csv file .

    :param via_file: path to the via csv file
    :type via_file: str
    :return: path to the clean csv file
    :rtype: str
    """
    out_file = via_file[:-4] + '_clean.csv'
    with open(via_file[:-4] + '_clean.csv', 'w') as cleanfile:
        with open(via_file, 'r') as file:
            lines = file.readlines()
            _ = cleanfile.writelines(lines[0].replace(',', '|'))
            for line in lines[1:]:
                curly_brackets = 0
                for character in line[1:-2]:
                    if character == '{':
                        curly_brackets += 1
                        _ = cleanfile.write(character)
                    elif character == '}':
                        curly_brackets -= 1
                        _ = cleanfile.write(character)
                    else:
                        if curly_brackets == 0:
                            _ = cleanfile.write(character.replace(',', '|'))
                        else:
                            _ = cleanfile.write(character)
                _ = cleanfile.write("\n")
    return out_file


def get_image_names(clean_via_file: str) -> 'list[str]':
    """
    Returns a list of the image file names annotated in the cleaned via file given as input.

    :param clean_via_file: path to the via csv file cleaned with the clean_via_file function
    :type clean_via_file: str
    :return: list of the image file names
    :rtype: list[str]
    """
    images_list = set()
    with open(clean_via_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        for row in reader:
            if row[0].endswith('.jpg'):
                images_list.add(row[0])
    images_list = list(images_list)
    return images_list


def cut_overlength(center: float, length: float) -> 'tuple[float,float]':
    """
    Trims a box in case it goes outside of the image and returns
    the new coordinates of the center and length.

    :param center: coordinate (x or y) of the box center
    :type center: float
    :param length: width or height of the box
    :type length: float
    :return: new values of the center and of the length
    :rtype: tuple[float, float]
    """
    if (center - (length / 2) < 0) and (center + (length / 2) <= 1):
        overlength = (length / 2) - center
        new_center = center + overlength / 2
        new_length = length - overlength
    elif (center - (length / 2) >= 0) and (center + (length / 2) > 1):
        overlength = center + (length / 2) - 1
        new_center = center - overlength / 2
        new_length = length - overlength
    elif (center - (length / 2) < 0) and (center + (length / 2) > 1):
        new_center = 0.5
        new_length = 1
    else:
        new_center = center
        new_length = length
    return (new_center, new_length)


def convert_via_to_yolo(via_file: str, out_dir: str, blurring: bool = False, noise: bool = False):
    """
    Downloads the images annotated in the via csv file and creates one txt file / image
    with the class and coordinates of the boxes as per yolo input format :
    <object-class> <x> <y> <width> <height>

    :param via_file: path to the via csv file
    :type via_file: str
    :param out_dir: path to the outputs directory
    :type out_dir: str
    :param blurring: if True, original data is augmented with a random blurring
    :type blurring: bool
    :param noise: if True, original data is augmented with a random noise
    :type noise: bool
    """
    via_clean = clean_via_file(via_file)
    images = get_image_names(via_clean)

    with(open("conf.json", "r")) as f:
        conf = json.load(f)
    if conf["bucket_standardized"] and conf['profile']:
        images_bucket = Bucket(conf["bucket_standardized"], conf['profile'])
    else:
        print("edit config file and add missing arguments")

    images_boxes = pd.read_csv(via_clean, sep='|', header=0)
    images_boxes = images_boxes[['filename', 'region_shape_attributes', 'region_attributes']]

    class_mapping = {'6': 0, '1': 1, '2': 0, '3': 3, '4': 2}

    for image in images:
        # Download image from s3 :
        images_bucket.download(image, out_dir)
        if blurring == True:
            image_array = image_read(os.path.join(out_dir, image))
            blurring_box_width = random.choice([7, 9, 11, 13])
            blurring_box_height = random.choice([7, 9, 11, 13])
            image_b = add_blurring(image_array, blurring_box_width, blurring_box_height)
            image_b_name = image[:-4] + '_b' + '.jpg'
            image_write(image_b, os.path.join(out_dir, image_b_name))
        if noise == True:
            image_array = image_read(os.path.join(out_dir, image))
            noise_var = random.randint(1000, 10000)
            image_n = add_noise(image_array, noise_var)
            image_n_name = image[:-4] + '_n' + '.jpg'
            image_write(image_n, os.path.join(out_dir, image_n_name))

        # Get image size :
        height, width, _ = get_img_shape(os.path.join(out_dir, image))
        # Filter DataFrame to get the rows of this image :
        boxes = images_boxes[images_boxes['filename'] == image]
        # Prepare path of the txt files :
        txt_file = os.path.join(out_dir, image[:-3] + 'txt')

        for index, box in boxes.iterrows():
            box_shape = eval(box.region_shape_attributes.strip('"').replace('""""', '"'))
            box_attributes = eval(box.region_attributes.strip('"').replace('""""', '"'))
            box_class = class_mapping[box_attributes['type']]
            x = round(
                (max(box_shape['all_points_x']) + min(box_shape['all_points_x'])) / (2 * width), 6)
            y = round(
                (max(box_shape['all_points_y']) + min(box_shape['all_points_y'])) / (2 * height), 6)
            box_width = round(
                (max(box_shape['all_points_x']) - min(box_shape['all_points_x'])) / (width), 6)
            box_height = round(
                (max(box_shape['all_points_y']) - min(box_shape['all_points_y'])) / (height), 6)

            x, box_width = cut_overlength(x, box_width)
            y, box_height = cut_overlength(y, box_height)
            with open(txt_file, 'a') as txt:
                _ = txt.write(f'{box_class} {x} {y} {box_width} {box_height}\n')

        if blurring == True:
            blurred_txt_path = os.path.join(out_dir, image[:-4] + '_b.txt')
            copyfile(txt_file, blurred_txt_path)

        if noise == True:
            noised_txt_path = os.path.join(out_dir, image[:-4] + '_n.txt')
            copyfile(txt_file, noised_txt_path)


def check_yolo_txt(image_path: str):
    """
    Opens a new window showing the image and the boxes found in the corresponding yolo txt file.
    yolo txt file must be in the same folder as the image.

    :param image_path: path to the image
    :type image_path: str
    """
    txt_path = image_path[:-3] + 'txt'
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
    with(open(txt_path, "r")) as txt:
        for line in txt:
            rect_class = int(line.split(' ')[0])
            x = float(line.split(' ')[1])
            y = float(line.split(' ')[2])
            width = float(line.split(' ')[3])
            height = float(line.split(' ')[4])
            x1 = int((x - (width / 2)) * 500)
            y1 = int((y - (height / 2)) * 500)
            x2 = int((x + (width / 2)) * 500)
            y2 = int((y + (height / 2)) * 500)
            cv2.rectangle(resized, (x1, y1), (x2, y2), (
                (rect_class == 1) * 255, (rect_class == 2) * 255, (rect_class == 3) * 255), 1)

    cv2.imshow('Image', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# check_yolo_txt('tools/yolo/data/eabef3a8-ad7e-4a4f-b035-a58f18d8393f.jpg')
