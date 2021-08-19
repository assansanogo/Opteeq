import json
from copy import deepcopy
from typing import Iterator, Tuple, Union

from google.cloud.vision_v1.types.image_annotator import AnnotateImageResponse
from tqdm import tqdm

from .storage import storage_factory
from .structure.default import default
from .text_box import textbox_factory


class ViaConverter:
    def __init__(self, cloud_name: str, local: bool, source_path: str, profile: str = "default"):
        """
        Object to convert text detection from cloud to via json.
        Possible to use with file store locally or in aws bucket.
        Possible to use vision from gcloud or rekognition from aws for text detection.

        :param cloud_name: cloud provider name "aws" for aws, "gcloud for gcloud
        :param source_path: path of the image folder or bucket name
        :param local: boolean False use bucket, true local storage
        :param profile: Choose AWS CLI profile if more than 1 are set up
        """
        self.storage = storage_factory(local, source_path, profile=profile)
        self.provider = textbox_factory(cloud_name, profile=profile)

    def request_generator(self, list_image: list) -> \
            Iterator[Tuple[str, Union[dict, AnnotateImageResponse]]]:
        """
        Return an iterator which return a tuple:
                                - name of image
                                - result of the cloud text detection (aws or gcloud)

        For each listed file. The file use can be store locally or in aws bucket.

        :param list_image: list all image to annotate.
        :return: iterator, tuple name of image and google vision response  for gcloud
            or python dict for aws.
        """

        for file in tqdm(list_image, desc="current batch", leave=False):
            response = self.provider.request(self.storage.read(file))
            yield file, response

    def via_json(self, list_image: list) -> dict:
        """
        Build dict for VGG Image Annotator with annotation from cloud.

        :param list_image: list all image to annotate.
        :return: dict with via format
        """
        output = deepcopy(default)
        for file_name, response in self.request_generator(list_image):
            # part for add an image
            output["_via_image_id_list"].append(file_name)
            image = {"file_attributes": {}, "filename": file_name,
                     "regions": self.provider.process_region(response), "size": 1}
            output["_via_img_metadata"][file_name] = image.copy()
        return output

    def via_json_save_locally(self, list_file: Union[None, list] = None) -> None:
        """
        Save locally in json the result of the via json for all the image in folder.

        :param list_file: list all image to annotate. if None use all file in buckets or folder.
        """
        if not list_file:
            list_file = self.storage.list_files()
        with open("via.json", "w") as f:
            json.dump(self.via_json(list_file), f)


if __name__ == '__main__':
    # via_converter = ViaConverter(cloud_name="gcloud", local=True, source_path="../../image")
    # gcloud localy
    # ViaConverter("gcloud", True, "../image").via_json_save_locally()
    # gcloud with s3 storage
    # ViaConverter("gcloud", False, "dsti-lab-leo").via_json_save_locally()
    # aws rekognition with local
    # ViaConverter("aws", True, "../image").via_json_save_locally()
    # aws rekognition with bucket storage
    # ViaConverter("aws", False, "dsti-lab-leo").via_json_save_locally()
    # TODO add unit test with moto for boto3
    pass
