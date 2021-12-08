"""
prediction without darknet with only opencv
more information: https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# this import is pure python code this function don't depend on libdarknet.so
from darknet import draw_boxes, class_colors
from predict import merge_bbox, extract_text
from predict import point2bbox


def get_network(cfg_path: str, weights_path: str) -> tuple[cv2.dnn_Net, list]:
    """
    build darknet network with darknet config file and weights

    :param cfg_path: path of the config darknet file
    :param weights_path: path of the weights
    :return: net and output layer
    """
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, ln


def get_classes(classes_name_path: str) -> tuple[list[str], dict]:
    """
    get classes from darknet classe name file

    :param classes_name_path: path of the classe name file
    :return: list with classe name and colors code
    """
    classes = open(classes_name_path).read().strip().split('\n')
    np.random.seed(42)
    colors = class_colors(classes)
    return classes, colors


def load_image(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    load image and process it to be used in open cv network

    :param path: path of the image
    :return: img and blob for opencv
    """

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (602, 602), interpolation=cv2.INTER_LINEAR)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (602, 602), swapRB=True, crop=False)
    return img, blob


def process_image(image_path: str, cfg_path: str, weights_path: str, classes_name_path: str) -> \
        tuple[np.ndarray, dict]:
    """
    process a ticket with yolo using open cv (no darknet)

    :param image_path: path of the image
    :param cfg_path:  path of the config darknet file
    :param weights_path: weights_path: path of the weights
    :param classes_name_path: path of the classe name file
    :return: image with boxes and the text of each classes in dict
    """

    result = {}
    img, blob = load_image(image_path)

    classes, colors = get_classes(classes_name_path)

    net, ln = get_network(cfg_path, weights_path)

    net.setInput(blob)

    outputs = net.forward(ln)
    outputs = np.vstack(outputs)
    detection = detection_darknet_format(img, outputs, 0.5, classes)
    detections_merge = merge_bbox(detection, "PLACE")
    for i in detections_merge:
        result[i[0]] = extract_text(i, img)
    img = draw_boxes(detection, img, colors)
    return img, result


def detection_darknet_format(img: np.ndarray, outputs: np.ndarray, conf: float,
                             classes: list) -> list:
    """
    convert opencv detection in yolo darknet detection format

    :param img: image
    :param outputs: outputs of opencv nn
    :param conf: threshold
    :param classes: list of all classes
    :return: list with yolo detection format ( tupple (classe, probability, bbox))
    """
    H, W = img.shape[:2]

    boxes = []
    confidences = []
    classIDs = []

    detections = []

    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > conf:
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w // 2), int(y - h // 2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, conf - 0.1)
    if len(indices) > 0:
        for i in indices.flatten():
            if 'dont_show' != classes[classIDs[i]].split(" ")[0]:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                detections.append(
                    (classes[classIDs[i]], confidences[i], point2bbox(x, y, x + w, y + h)))
    return detections


if __name__ == '__main__':
    weights_path = 'docker/data/yolov4-custom_best.weights'
    input_path = '/home/souff/darknet/obj/0a865cb9-0afc-4791-bec1-c88a2a07b3ee.jpg'
    datafile_path = 'obj.data'
    classes_name = 'docker/data/obj.names'
    cfg_path = 'docker/data/yolov4-custom.cfg'
    input_file_path = 'docker/data/validation.txt'

    img, result = process_image(input_path, cfg_path, weights_path, classes_name)
    plt.imshow(img)
    plt.show()
