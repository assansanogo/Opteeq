import cv2
import numpy as np
import os


def resize_and_pad(img: np.ndarray, desired_size: int = 180) -> np.ndarray:
    """Resizes an image to a square image of the desired size, with a black pad.

    :param img: Image array
    :type img: np.ndarray
    :param desired_size: Final image size, defaults to 180
    :type desired_size: int, optional
    :return: Resized image array
    :rtype: np.ndarray
    """

    old_size = img.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=color)
    return new_img


def image_read(file: str) -> np.ndarray:
    """
    Reads a single image into a numpy array using OpenCV
    :param file: path to the image
    :type img: str
    :return: Image array
    :rtype: np.ndarray
    """
    img = cv2.imread(file)
    return img


def rotate_image(img: np.ndarray, angle: int = 90) -> np.ndarray:
    """Rotates an image with the desired angle.

    :param img: Image array
    :type img: np.ndarray
    :param angle: Angle of rotation in degrees, defaults to 90
    :type angle: int, optional
    :return: Rotated image array
    :rtype: np.ndarray
    """
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


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


def prepare_training_images(in_folder: str, out_folder: str) -> list:
    """Uses all image files in a folder to get 4 images / file that can be used to train the model.
    The N images in the original folder must all be in the normal orientation.
    The 4xN processed images in the output folder are squared with size 180 and rotated in 4
    orientations (0, 90, 180, and 270 degrees). The last 3 characters of the file names indicates the
    orientation : '000', '090', '180' or '270'.

    :param in_folder: path to the folder with original images
    :type in_folder: str
    :param out_folder: path to the folder to upload the processed images
    :type out_folder: str
    :return: a List of the files that were not processed
    :rtype: List of str
    """
    not_worked = []

    for filename in os.listdir(in_folder):

        try:
            file_path = os.path.join(in_folder, filename)
            filename_base = filename.split('.')[0]
            prepare_1_image(file_path, filename_base, out_folder)
        except:
            not_worked.append(filename)

    return not_worked
