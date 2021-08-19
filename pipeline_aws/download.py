"""
Part D cf schema
"""
import json
import os

from tqdm import tqdm

from tools.aws.awsTools import Bucket


def download(annotator_name: str, bucket_json: str, bucket_image,
             local_storage: str = "download", profile: str = "default") -> None:
    """
    download via json and image which aren't store in local. Edit json with path image folder.

    :param annotator_name: name of the user
    :param bucket_json: bucket where json via are stored
    :param bucket_image: bucket where image are stored
    :param local_storage: path of the local storage where json and image are stored
    :param profile: Choose AWS CLI profile if more than 1 are set up
    """
    bucket = Bucket(bucket_image, profile)
    json_set = download_via_json(annotator_name, bucket_json, local_storage, profile)
    for json_name in tqdm(json_set, desc="download json", leave=False):
        json_path = os.path.join(local_storage, "json", json_name)
        with open(json_path, "r") as file:
            json_via = json.load(file)
            path_folder = os.path.join(local_storage, "image", json_name.split(".")[0])
            os.mkdir(path_folder)
            for key in tqdm(json_via["_via_img_metadata"].keys(), desc="image/curent json",
                            leave=False):
                bucket.download(key, path_folder)
        # auto set image path folder
        with open(json_path, "w") as file:
            json_via["_via_settings"]["core"]["default_filepath"] = os.path.join(
                os.path.abspath(path_folder), "")
            json.dump(json_via, file)


def download_via_json(annotator_name: str, bucket_name: str,
                      local_storage: str = "download", profile: str = "default") -> set:
    """
    download the via json which contains the right annotator name from the bucket.
    Check local storage, if a file is already store, don't download.

    :param annotator_name: name of the user
    :param bucket_name: bucket where json via are stored
    :param local_storage: path of the local storage where json via are stored
    :param profile: Choose AWS CLI profile if more than 1 are set up
    :return: set of json download
    """

    bucket = Bucket(bucket_name, profile)
    # use set for set operation
    file_bucket = set(bucket.list_object_search_key(annotator_name))
    files_local = set(os.listdir(os.path.join(local_storage, "json")))
    to_download = file_bucket - files_local
    for key in to_download:
        bucket.download(key, os.path.join(local_storage, "json"))
    return to_download


if __name__ == '__main__':
    with(open("conf.json", "r")) as f:
        conf = json.load(f)
    if conf["user"] and conf["start"] and conf["bucket_initial_annotation"] and \
            conf["bucket_standardized"]:
        download(conf["user"], conf["bucket_initial_annotation"], conf["bucket_standardized"],
                 profile=conf["profile"])
    else:
        print("edit config and add missing argument")
