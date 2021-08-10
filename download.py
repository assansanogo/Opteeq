"""
Part D cf schema
"""
from tools.aws.awsTools import Bucket
import os
import json


def main(annotator_name: str, bucket_json: str, bucket_image,
         local_storage: str = "download") -> None:
    """
    download via json and image which aren't store in local.

    :param annotator_name: name of the user
    :param bucket_json: bucket where json via are stored
    :param bucket_image: bucket where image are stored
    :param local_storage: path of the local storage where json and image are stored
    """
    bucket = Bucket(bucket_image)
    json_set = download_via_json(annotator_name, bucket_json, local_storage)
    for json_name in json_set:
        with open(os.path.join(local_storage, "json", json_name), "r") as f:
            json_via = json.load(f)
            path_folder = os.path.join(local_storage, "image", json_name.split(".")[0])
            os.mkdir(path_folder)
            for key in json_via["_via_img_metadata"].keys():
                bucket.download(key, path_folder)


def download_via_json(annotator_name: str, bucket_name: str,
                      local_storage: str = "download") -> set:
    """
    download the via json which contains the right annotator name from the bucket.
    Check local storage, if a file is already store, don't download.

    :param annotator_name: name of the user
    :param bucket_name: bucket where json via are stored
    :param local_storage: path of the local storage where json via are stored
    :return: set of json download
    """

    bucket = Bucket(bucket_name)
    # use set for set operation
    file_bucket = set(bucket.list_object_search_key(annotator_name))
    files_local = set((os.listdir(local_storage)))
    to_download = file_bucket - files_local
    for key in to_download:
        bucket.download(key, os.path.join(local_storage, "json"))
    return to_download


if __name__ == '__main__':
    with(open("conf.json", "r")) as f:
        conf = json.load(f)
    if conf["user"] and conf["start"] and conf["bucket_initial_annotation"] and conf[
        "bucket_standardized"]:
        main(conf["user"], conf["bucket_initial_annotation"], conf["bucket_standardized"])
    else:
        print("edit config and add missing argument")
