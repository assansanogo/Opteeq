import os
import random

import cv2

from tools.image.imageTools import resize_and_pad, rotate_image, image_read, add_blurring, add_noise


def prepare_1_image(file: str, resolution: int, filename: str, out_folder: str):
    """Resizes an image, add padds to get a square image, rotates it 3 times, and adds blurring and
    noise to get 12 images that can be used to train a model.

    :param file: path to the original image
    :type file: str
    :param resolution: width/height of the output images in pixels
    :type resolution: int
    :param filename: filename base of the processed images
    :type filename: str
    :param out_folder: path to the folder to upload the 12 processed images
    :type out_folder: str
    """
    img = image_read(file)
    # Initializing random parameters for blurring and noising functions
    box_width = 3
    box_height = 3
    noise_vars = [random.randint(9, 100) for i in range(4)]

    resized_img = resize_and_pad(img, desired_size=resolution)

    for index, angle in enumerate(['000', '090', '180', '270']):
        if index != 0:
            resized_img = rotate_image(resized_img)
        resized_img_b = add_blurring(resized_img, box_width, box_height)
        resized_img_n = add_noise(resized_img, noise_vars[index])
        cv2.imwrite(os.path.join(out_folder, f'{filename}_{angle}.jpg'), resized_img)
        cv2.imwrite(os.path.join(out_folder, f'{filename}_b_{angle}.jpg'), resized_img_b)
        cv2.imwrite(os.path.join(out_folder, f'{filename}_n_{angle}.jpg'), resized_img_n)


def prepare_training_images(resolution: int, in_folder: str,
                            out_folder: str) -> 'tuple[int, list[str]]':
    """
    Uses all image files in a folder to get 12 images / file that can be used to train a model.
    The N images in the original folder must all be in the normal orientation.
    The 12xN processed images in the output folder are squared with the desired resolution,
    rotated in 4 orientations (0, 90, 180, and 270 degrees) and with additional noise or blurring.
    The last 3 characters of the file names indicate the orientation : '000','090', '180' or '270'.

    :param resolution: width/height of the output images in pixels
    :type resolution: int
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
            prepare_1_image(file_path, resolution, filename_base, out_folder)
            file_counter += 12
        except:
            not_worked.append(filename)

    return (file_counter, not_worked)


def prepare_testing_images(resolution: int, in_folder: str,
                           out_folder: str) -> 'tuple[int, list[str]]':
    """
    Resizes and adds pads to the image files in a folder and turns some of them to obtain a balanced
    dataset that can be used to test a model.
    The N images in the original folder must all be in the normal orientation.
    The N processed images in the output folder are squared with the desired resolution,
    and rotated at 0, 90, 180, or 270 degrees.
    The last 3 characters of the file names indicate the orientation : '000','090', '180' or '270'.

    :param resolution: width/height of the output images in pixels
    :type resolution: int
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
            img = image_read(file_path)
            resized_img = resize_and_pad(img, desired_size=resolution)
            nb_rotations = file_counter % 4
            angle = str(nb_rotations * 90).zfill(3)
            for i in range(nb_rotations):
                resized_img = rotate_image(resized_img)
            cv2.imwrite(os.path.join(out_folder, f'{filename}_{angle}.jpg'), resized_img)
            file_counter += 1
        except:
            not_worked.append(filename)

    return (file_counter, not_worked)


if __name__ == '__main__':
    in_folder_parameter = 'C:/Users/johan/OneDrive - Data ScienceTech Institute/Data Science/Python/Projects/Assan-opteeq/val_raw'
    out_folder_parameter = 'C:/Users/johan/OneDrive - Data ScienceTech Institute/Data Science/Python/Projects/Assan-opteeq/val'
    resolution = 224
    file_counter, not_worked = prepare_training_images(resolution, in_folder_parameter,
                                                       out_folder_parameter)

    print(f'{file_counter} training images uploaded')

    if not_worked != []:
        print('The following files could not be processed :')
        print(str(not_worked))

    in_folder_parameter = 'C:/Users/johan/OneDrive - Data ScienceTech Institute/Data Science/Python/Projects/Assan-opteeq/test_raw'
    out_folder_parameter = 'C:/Users/johan/OneDrive - Data ScienceTech Institute/Data Science/Python/Projects/Assan-opteeq/test'
    resolution = 224
    file_counter, not_worked = prepare_testing_images(resolution, in_folder_parameter,
                                                      out_folder_parameter)

    print(f'{file_counter} testing images uploaded')

    if not_worked != []:
        print('The following files could not be processed :')
        print(str(not_worked))
