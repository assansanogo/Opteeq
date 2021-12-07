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


def detection(weights: str, input_image: str, datafile: str, cfg: str) -> tuple[np.ndarray, list]:
    """
    Use yolo to detect classes in the image.
    Work only on 1 image with its path.
    Important you need to have create libdarknet.so at the compilation.
    return the detection with bouding boxes and the image (with boxes on it).

    :param weights: path of the best weights file (after trainning)
    :param input_image: path of an image
    :param datafile: path of yolo datafile (example obj.data) use for the trainning
    :param cfg: path of yolo config file (example yolov4-custom.cfg) use for trainning
    """
    network, class_names, class_colors = load_network(cfg, datafile, weights, 1)
    output_image, detections = image_detection_dont_show(
        input_image,
        network,
        class_names,
        class_colors, thresh=0.5
    )
    return output_image, detections


def detection_several_image(weights: str, input_file: str, datafile: str, cfg: str) -> list:
    """
    Use yolo to detect classes in the image.
    Work on a set of image with a txt file which contains all the path of images.
    Important you need to have create libdarknet.so at the compilation.
    return a dict with the dectection for each image and save the image with bouding boxes in data.

    :param weights: path of the best weights file (after trainning)
    :param input_file: path of an image
    :param datafile: path of yolo datafile (example obj.data) use for the trainning
    :param cfg: path of yolo config file (example yolov4-custom.cfg) use for trainning
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
            image, detection, = image_detection_dont_show(i,
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


def extract_text(detection: list, image: np.ndarray) -> tuple[tuple, str]:
    """
    Extract text from an image for a given bouding box of yolo
    return tuple with yolo detection and string of the text

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
    return (left, top, right, bottom), text


def recipe_process(weights: str, input: str, datafile: str, cfg: str) -> tuple[
    np.ndarray, np.ndarray, dict]:
    """
    process a ticket with yolo
    return image anotated with yolo, image with boxes marged and the text of each classes in dict

    :param weights: path of the best weights file (after trainning)
    :param input: path of an image
    :param datafile: path of yolo datafile (example obj.data) use for the trainning
    :param cfg: path of yolo config file (example yolov4-custom.cfg) use for trainning
    """
    result = {}
    points = []
    image = cv2.imread(input)
    image = cv2.resize(image, (602, 602),
                       interpolation=cv2.INTER_LINEAR)
    annotated_image, detections = detection(weights, input, datafile, cfg)
    detections = merge_bbox(detections, "PLACE")
    for i in detections:
        point, result[i[0]] = extract_text(i, image)
        points.append(point)
    for i in points:
        left, top, right, bottom = i
        image = cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    return image, annotated_image, result


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


def image_detection_dont_show(image_path, network, class_names, class_colors, thresh):
    """
    image detection from darknet wrapper, with support of dont_show in classe name like in bash yolo
    (dont_show classe are ignored).


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
    image = draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


if __name__ == '__main__':
    weights_path = 'recipe/yolov4-custom_best.weights'
    input_path = 'obj/0a865cb9-0afc-4791-bec1-c88a2a07b3ee.jpg'
    datafile_path = 'recipe/obj.data'
    cfg_path = 'recipe/yolov4-custom.cfg'
    input_file_path = 'recipe/validation.txt'

    # detect with path
    image, annotated_image, result = recipe_process(weights_path, input_path, datafile_path,
                                                    cfg_path)
    plt.imshow(annotated_image)
    plt.show()
