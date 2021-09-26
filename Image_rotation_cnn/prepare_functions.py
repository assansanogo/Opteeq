from tools.image.imageTools import resize_and_pad, rotate_image, image_read, add_blurring, add_noise
import random
import cv2
import os


def prepare_1_image(file: str, filename: str, out_folder: str):
    """Resizes an image, rotates it 3 times, adds blurring and noise to get 12 images that
    can be used to train a model.

    :param file: path to the original image
    :type file: str
    :param filename: filename base of the processed images
    :type filename: str
    :param out_folder: path to the folder to upload the 12 processed images
    :type out_folder: str
    """
    img = image_read(file)
    # Initializing random parameters for blurring and noising functions
    box_widths = random.choices([3,5,7,9],k=4)
    box_heights = random.choices([3,5,7,9],k=4)
    noise_vars = [random.randint(25,1000) for i in range(4)]
    
    resized_img = resize_and_pad(img, desired_size=180)
    
    for index, angle in enumerate(['000', '090', '180', '270']):
        if index != 0:
            resized_img = rotate_image(resized_img)
        resized_img_b = add_blurring(resized_img,box_widths[index],box_heights[index])
        resized_img_n = add_noise(resized_img, noise_vars[index])
        cv2.imwrite(os.path.join(out_folder, f'{filename}_{angle}.jpg'), resized_img)
        cv2.imwrite(os.path.join(out_folder, f'{filename}_b_{angle}.jpg'), resized_img_b)
        cv2.imwrite(os.path.join(out_folder, f'{filename}_n_{angle}.jpg'), resized_img_n)

    #resized_img = resize_and_pad(img, desired_size=180)
    #resized_img_b = add_blurring(resized_img,box_widths[0])
    #rotated_img_90 = rotate_image(resized_img)
    #rotated_img_180 = rotate_image(rotated_img_90)
    #rotated_img_270 = rotate_image(rotated_img_180)
    #os.path.join(out_folder, f'{filename}_000.jpg')
    #cv2.imwrite(os.path.join(out_folder, f'{filename}_000.jpg'), resized_img)
    #cv2.imwrite(os.path.join(out_folder, f'{filename}_090.jpg'), rotated_img_90)
    #cv2.imwrite(os.path.join(out_folder, f'{filename}_180.jpg'), rotated_img_180)
    #cv2.imwrite(os.path.join(out_folder, f'{filename}_270.jpg'), rotated_img_270)

def prepare_training_images(in_folder: str, out_folder: str) -> 'tuple[int, list[str]]':
    """Uses all image files in a folder to get 12 images / file that can be used to train a model.
    The N images in the original folder must all be in the normal orientation.
    The 12xN processed images in the output folder are squared with size 180, rotated in 4
    orientations (0, 90, 180, and 270 degrees) and with additional noise or blurring.
    The last 3 characters of the file names indicate the orientation : '000',
     '090', '180' or '270'.

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
            file_counter += 12
        except:
            not_worked.append(filename)

    return (file_counter, not_worked)



prepare_1_image('Image_rotation_cnn/Original/20210913_122444.jpg', 'bob', 'Image_rotation_cnn/Original/')

bob, listbob = prepare_training_images('Image_rotation_cnn/Original/', 'Image_rotation_cnn/Original/')