import base64
import json
import time

import cv2

from yolo.predict import recipe_process
from yolo.predict_opencv import process_image


def handler(event, context):
    path_weight = 'yolo/docker/data/yolov4-custom_best.weights'
    path_names = 'yolo/docker/data/obj.names'
    path_data = 'yolo/docker/data/obj.data'
    path_config = 'yolo/docker/data/yolov4-custom.cfg'

    start_time = time.time()
    opencv = event.get("opencv", None)
    if 'image' in event:
        with open("/tmp/saved_img.png", "wb") as f:
            f.write(base64.b64decode(event["image"]))
        image_path = '/tmp/saved_img.png'
    # use default image for demo
    else:
        image_path = 'yolo/docker/data/0a865cb9-0afc-4791-bec1-c88a2a07b3ee.jpg'
    if opencv:
        image, detection = process_image(image_path, path_config, path_weight, path_names)
    else:
        image, detection = recipe_process(path_weight, image_path, path_data, path_config)
    _, buffer_img = cv2.imencode('.jpg', image)
    image64 = base64.b64encode(buffer_img).decode("utf-8")
    final_time = time.time() - start_time
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": detection,
            "image64": image64,
            "time": final_time
        }),
    }
