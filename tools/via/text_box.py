"""
Generating image box with cloud service.
"""
import io
from abc import ABC, abstractmethod
from typing import Union

from PIL import Image
from google.cloud import vision
from google.cloud.vision_v1.types.image_annotator import AnnotateImageResponse

from tools.aws.awsTools import Rekognition


class TextBox(ABC):
    """
    Interface for Generating box with cloud service on image.
    """

    @abstractmethod
    def request(self, img: bytes) -> Union[AnnotateImageResponse, dict]:
        """
        Make text detection request on cloud provider.

        :param img: image in bytes
        :return: cloud response
        """
        pass

    @abstractmethod
    def process_region(self, response: Union[AnnotateImageResponse, dict]) -> list:
        """
        convert text region detection response from cloud provider in a standardized format.
        :param response: response from cloud provider
        :return: list
        """
        pass


class GcloudVision(TextBox):
    """
    text detection with google cloud vision.
    """

    def __init__(self, **kwargs):
        self.client = vision.ImageAnnotatorClient()

    def request(self, img: bytes) -> AnnotateImageResponse:
        """
        Make text detection request with google vision.

        :param img: image in bytes
        :return: google cloud vision annotate image response object.
        """
        image = vision.Image({"content": img})
        return self.client.text_detection(image=image)

    def process_region(self, response: AnnotateImageResponse) -> list:
        """
        convert vision response region in standardized format.

        :param response: google vision annotate image response object.
        :return: Region standardized, python list.
        """
        region = []
        for box in response.text_annotations:
            region.append(
                {"region_attributes": {"Text": box.description, "type": "6"}, "shape_attributes": {
                    "all_points_x": [point.x for point in box.bounding_poly.vertices],
                    "all_points_y": [point.y for point in box.bounding_poly.vertices],
                    "name": "polygon"}})
        return region


class AwsRekognition(Rekognition, TextBox):
    """
    text detection with aws rekognition
    """

    def __init__(self, region: str = "eu-west-1", profile: str = "default"):
        """
        :param region: aws service region
        :param profile: Choose AWS CLI profile if more than 1 are set up
        """
        super().__init__(profile=profile, region=region)

    def request(self, img: bytes) -> dict:
        """
        Make text detection request with aws rekognition.

        :param img: image in bytes
        :return: aws rekognition response.
        """
        response = self.get_text(img)
        image = Image.open(io.BytesIO(img))
        width, height = image.size
        response["width"] = width
        response["height"] = height
        return response

    def process_region(self, response: dict) -> list:
        """
        convert aws rekognition response region in standardized format.

        :param response: aws rekognition response
        :return: Region standardized, python list.
        """
        region = []
        width = response["width"]
        height = response["height"]
        for box in response["TextDetections"]:
            region.append(
                {"region_attributes": {"Text": box["DetectedText"], "type": "6"},
                 "shape_attributes": {
                     "all_points_x": [point["X"] * width for point in
                                      box["Geometry"]["Polygon"]],
                     "all_points_y": [point["Y"] * height for point in
                                      box["Geometry"]["Polygon"]],
                     "name": "polygon"}})
        return region


def textbox_factory(cloud_name: str, **kwargs) -> TextBox:
    """
    factory to create the textbox object.

    :param cloud_name: cloud provider name "aws" for aws, "gcloud for gcloud.
    :keyword Arguments:
        :key region (str): aws service region
        :key profile (str): Choose AWS CLI profile if more than 1 are set up
    :return: TextBox object
    """
    textbox = {"aws": AwsRekognition, "gcloud": GcloudVision}
    return textbox[cloud_name](**kwargs)
