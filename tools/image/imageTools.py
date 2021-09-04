import numpy as np
import cv2

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
    """Reads a single image into a numpy array using OpenCV.

    :param file: path to the image
    :type file: str
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

def get_img_shape(file: str) -> 'tuple[int, int, int]':
    """Returns the shape of an image in a tuple (height, width, #channels).

    :param file: path to the image
    :type file: str
    :return: (height, width, #channels) of the image
    :rtype: tuple[int, int, int]
    """
    img = cv2.imread(file)
    try:
        return img.shape
    except AttributeError:
        print(f'error to get the shape of file {file}')
        return (None, None, None)