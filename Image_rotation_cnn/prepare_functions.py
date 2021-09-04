from tools.image.imageTools import resize_and_pad, rotate_image, image_read
import cv2
import numpy as np
import os


def prepare_1_image(file: str, filename: str, out_folder: str):
    """Resizes an image and rotate it 3 times to get 4 images that can be used to train the model.

    :param file: path to the original image
    :type file: str
    :param filename: filename base of the 4 processed images
    :type filename: str
    :param out_folder: path to the folder to upload the 4 processed images
    :type out_folder: str
    """
    img = image_read(file)

    resized_img = resize_and_pad(img, desired_size=180)
    rotated_img_90 = rotate_image(resized_img)
    rotated_img_180 = rotate_image(rotated_img_90)
    rotated_img_270 = rotate_image(rotated_img_180)
    os.path.join(out_folder, f'{filename}_000.jpg')
    cv2.imwrite(os.path.join(out_folder, f'{filename}_000.jpg'), resized_img)
    cv2.imwrite(os.path.join(out_folder, f'{filename}_090.jpg'), rotated_img_90)
    cv2.imwrite(os.path.join(out_folder, f'{filename}_180.jpg'), rotated_img_180)
    cv2.imwrite(os.path.join(out_folder, f'{filename}_270.jpg'), rotated_img_270)

def prepare_training_images(in_folder: str, out_folder: str) -> 'tuple[int, list[str]]':
    """Uses all image files in a folder to get 4 images / file that can be used to train the model.
    The N images in the original folder must all be in the normal orientation.
    The 4xN processed images in the output folder are squared with size 180 and rotated in 4
    orientations (0, 90, 180, and 270 degrees). The last 3 characters of the file names indicates the
    orientation : '000', '090', '180' or '270'.

    :param in_folder: path to the folder with original images
    :type in_folder: str
    :param out_folder: path to the folder to upload the processed images
    :type out_folder: str
    :return: a Tuple with the number of images uploaded and a List of the files that were not processed
    :rtype: (int, List of str)
    """
    not_worked = []
    file_counter = 0
    for filename in os.listdir(in_folder):
        try:
            file_path = os.path.join(in_folder, filename)
            filename_base = filename.split('.')[0]
            prepare_1_image(file_path, filename_base, out_folder)
            file_counter += 4
        except:
            not_worked.append(filename)

    return (file_counter, not_worked)

