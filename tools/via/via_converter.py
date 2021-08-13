import io
import json
import os
from typing import Iterator, Tuple, Union
from tools.aws.awsTools import Bucket, Rekognition
from .structure.default import default
from google.cloud import vision
from google.cloud.vision_v1.types.image_annotator import AnnotateImageResponse
from tqdm import tqdm
from copy import deepcopy

class Local:
    """
    Read file from a given folder and store in memory.
    """

    def __init__(self, folder: str):
        """
        :param folder: folder path
        """
        self.folder = folder

    def read(self, file) -> bytes:
        """
        read file and store in memory

        :param file: file name
        :return: image bytes
        """
        with io.open(os.path.join(self.folder, file), 'rb') as image_file:
            return image_file.read()


def get_source(local: bool, source_path: str, profile: str = "default") -> Union[Bucket, Local]:
    """
    return the right source in function of local or not.

    :param local: boolean, true local storage false s3.
    :param source_path: file path or s3 bucket name.
    :param profile: Choose AWS CLI profile if more than 1 are set up
    :return: local object to read from local or Bucket object to read from s3.
    """
    if local:
        return Local(source_path)
    else:
        return Bucket(source_path, profile)


def request_generator(list_image: list, source_path: str, local: bool = False,
                      profile: str = "default") \
        -> Iterator[Tuple[str, AnnotateImageResponse]]:
    """
    Return an iterator which return a tuple:
                            - name of image
                            - result google vision API request

    For each listed file. Can work with local file or s3.

    :param list_image: list all image to annotate with google vision api.
    :param source_path: bucket name or path of the folder
    :param local: boolean False use bucket, true local storage
    :param profile: Choose AWS CLI profile if more than 1 are set up
    :return: iterator, tuple name of image and google vision response
    """
    client = vision.ImageAnnotatorClient()
    source = get_source(local, source_path, profile)
    for file in tqdm(list_image, desc="current batch", leave=False):
        content = source.read(file)
        image = vision.Image(content=content)
        yield file, client.text_detection(image=image)


def via_json(list_image: list, source_path: str, local: bool = False,
             profile: str = "default") -> dict:
    """
    Build dict for VGG Image Annotator with annotation from google cloud response.

    :param list_image: list all image to annotate with google vision api.
    :param source_path: path of the image folder or bucket name
    :param local: boolean False use bucket, true local storage
    :param profile: Choose AWS CLI profile if more than 1 are set up
    :return: dict with via format
    """
    output = deepcopy(default)
    for file_name, response in request_generator(list_image, source_path, local, profile):
        # part for add an image
        output["_via_image_id_list"].append(file_name)
        image = {"file_attributes": {}, "filename": file_name, "regions": [], "size": 1}
        for box in response.text_annotations:
            image["regions"].append(
                {"region_attributes": {"Text": box.description, "type": "6"}, "shape_attributes": {
                    "all_points_x": [point.x for point in box.bounding_poly.vertices],
                    "all_points_y": [point.y for point in box.bounding_poly.vertices],
                    "name": "polygon"}})
        output["_via_img_metadata"][file_name] = image.copy()
    return output


def via_json2(list_image: list, bucket: str, width=717, height=951) -> dict:
    """
    Build dict for VGG Image Annotator with annotation from aws rekognition.
    All image need to be upload on S3 and resize before.

    :param list_image: list all image to annotate with google vision api.
    :param bucket: bucket name
    :param width: image width
    :param height: image height
    :return: dict with via format
    """
    output = default
    rekognition = Rekognition(bucket)
    for key in tqdm(list_image, desc='current batch', leave=False):
        json_response = rekognition.get_text(key)
        # part for add an image
        output["_via_image_id_list"].append(key)
        image = {"file_attributes": {}, "filename": key, "regions": [], "size": 1}
        for box in json_response["TextDetections"]:
            image["regions"].append(
                {"region_attributes": {"type": 6, "Text": box["DetectedText"]},
                 "shape_attributes": {
                     "all_points_x": [point["X"] * width for point in box["Geometry"]["Polygon"]],
                     "all_points_y": [point["Y"] * height for point in box["Geometry"]["Polygon"]],
                     "name": "polygon"}})
        output["_via_img_metadata"][key] = image.copy()
    return output


def via_json_local(folder: str) -> None:
    """
    via json function apply on local file and create json.

    :param folder: path of the folder which contains image
    """
    list_file = os.listdir(folder)
    with open("via.json", "w") as f:
        json.dump(via_json(list_file, folder, local=True), f)


if __name__ == '__main__':
    via_json_local("../../image")
