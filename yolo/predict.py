"""
Use yolo python wrapper to make prediction
"""
import os.path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pytesseract import pytesseract

from darknet import load_network, bbox2points, network_width, network_height, make_image, \
    copy_image_from_bytes, detect_image, free_image, draw_boxes
from darknet_images import image_detection


def detection(weights: str, input_image: str, datafile: str, cfg: str) -> tuple[
    np.ndarray, list, dict]:
    """
    Use yolo to detect classes in the image.
    Work only on 1 image with its path.
    Important you need to have created libdarknet.so at the compilation.
    return the detection with bounding boxes, the image (with boxes on it) and class color.

    :param weights: path of the best weights file (after training)
    :param input_image: path of an image
    :param datafile: path of yolo datafile (example obj.data) use for the training
    :param cfg: path of yolo config file (example yolov4-custom.cfg) use for training
    """
    network, class_names, class_colors = load_network(cfg, datafile, weights, 1)
    output_image, detections = image_detection_dont_show(
        input_image,
        network,
        class_names, thresh=0.5
    )
    return output_image, detections, class_colors


def detection_several_image(weights: str, input_file: str, datafile: str, cfg: str) -> dict:
    """
    Use yolo to detect classes in the image.
    Work on a set of image with a txt file which contains all the path of images.
    Important you need to have created libdarknet.so at the compilation.
    return a dict with the detection for each image and save the image with bounding boxes in data.

    :param weights: path of the best weights file (after training)
    :param input_file: path of a txt file with several image path
    :param datafile: path of yolo datafile (example obj.data) use for the training
    :param cfg: path of yolo config file (example yolov4-custom.cfg) use for training
    """

    result = {}
    random.seed(3)  # deterministic bbox colors
    with open(input_file, 'r') as f:
        image_names = f.read().split('\n')
    network, class_names, class_colors = load_network(
        cfg,
        datafile,
        weights
    )
    for i in image_names:
        if os.path.isfile(i):
            image, detection, = image_detection(i,
                                                network,
                                                class_names,
                                                class_colors, thresh=0.5
                                                )
            cv2.imwrite(f"data/{i}", image)
            result[i.split('/')[-1]] = detection
    return result


def merge_bbox(detections: list, classe: str) -> list:
    """
    Merge boxe with the same classe label
    return yolo list detection

    :param detections: yolo detection list
    :param classe: name of the classe to merge
    """
    merge = [bbox2points(i[2]) for i in detections if i[0] == classe]
    left = min([i[0] for i in merge])
    top = min([i[1] for i in merge])
    right = max([i[2] for i in merge])
    bottom = max([i[3] for i in merge])
    result = [i for i in detections if i[0] != classe]
    result.append((classe, 0, point2bbox(left, top, right, bottom)))
    return result


def extract_text(detection: list, image: np.ndarray) -> str:
    """
    Extract text from an image for a given bounding box of yolo
    return string of the text

    :param detection: yolo detection
    :param image: image
    """
    if detection[0] == 'PLACE':
        configuration = ("-l eng")
    else:
        configuration = ("-l eng --oem 1 --psm 8")
    bbox = detection[2]
    left, top, right, bottom = bbox2points(bbox)
    img = image[top:bottom, left:right]
    text = pytesseract.image_to_string(img, config=configuration)
    text = text.replace('\n\x0c', '').replace('\n', ' ')
    return text


def recipe_process(weights: str, input: str, datafile: str, cfg: str) -> tuple[np.ndarray, dict]:
    """
    process a ticket with yolo
    return image with boxes merged and the text of each classes in dict

    :param weights: path of the best weights file (after training)
    :param input: path of an image
    :param datafile: path of yolo datafile (example obj.data) use for the training
    :param cfg: path of yolo config file (example yolov4-custom.cfg) use for training
    """
    result = {}
    image, detections, class_colors = detection(weights, input, datafile, cfg)
    detections_merge = merge_bbox(detections, "PLACE")
    for i in detections_merge:
        result[i[0]] = extract_text(i, image)
    # boxes add at the end to
    image = draw_boxes(detections, image, class_colors)
    return image, result


def point2bbox(xmin: int, ymin: int, xmax: int, ymax: int) -> tuple[float, float, float, float]:
    """
    convert an opencv point to bounding boxe of yolo
    return bounding box of yolo

    :param xmin: x of the top right corner point
    :param ymin: y of the top right corner point
    :param xmax: x of the bottom left corner point
    :param ymax: y of the bottom left corner point
    """
    x = (xmin + xmax) / 2
    w = xmax - xmin
    y = (ymin + ymax) / 2
    h = ymax - ymin
    return x, y, w, h


def image_detection_dont_show(image_path: str, network, class_names: dict, thresh: float) -> tuple[
    np.ndarray, list]:
    """
    image detection from darknet wrapper, with support of dont_show in classe name like in bash yolo
    (dont_show classe are ignored).

    :param image_path: path of the image
    :param network: model trained
    :param class_names: name of the different classe
    :param thresh:
    """
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = network_width(network)
    height = network_height(network)
    darknet_image = make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = detect_image(network, class_names, darknet_image, thresh=thresh)
    detections = [i for i in detections if i[0].split(" ")[0] != 'dont_show']  # add dont_show
    free_image(darknet_image)
    return cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB), detections


if __name__ == '__main__':
    weights_path = 'docker/data/yolov4-custom_best.weights'
    input_path = '/home/souff/darknet/obj/0a865cb9-0afc-4791-bec1-c88a2a07b3ee.jpg'
    datafile_path = 'obj.data'
    cfg_path = 'docker/data/yolov4-custom.cfg'
    input_file_path = 'docker/data/validation.txt'

    # detect with path
    image, result = recipe_process(weights_path, input_path, datafile_path,
                                   cfg_path)
    plt.imshow(image)
    plt.show()
