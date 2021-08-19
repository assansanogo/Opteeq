"""
Part A cf schema
"""
import json
import os
import pathlib

from tqdm import tqdm

from tools.aws.awsTools import Bucket


def upload(user: str, folder: str, start: int, bucket_raw: str, profile: str = "default"):
    """
    Upload all the files of folder in AWS bucket, rename file with user name and number auto increment

    :param user: user name
    :param folder: folder to upload
    :param start: start for auto increment
    :param bucket_raw: name bucket
    :param profile: Choose AWS CLI profile if more than 1 are set up
    :return: the final number auto increment in order to save it
    """
    bucket = Bucket(bucket_raw, profile)
    for filename in tqdm(os.listdir(folder), "upload"):
        bucket.upload(os.path.join(folder, filename),
                      f"{user}_{start}{pathlib.Path(filename).suffix}")
        start += 1
    return start


if __name__ == '__main__':
    # take all parameter in conf.json, allow to save the position since the last execution
    # and execute directly with python3 rename_upload.py in console
    with(open("conf.json", "r")) as f:
        conf = json.load(f)
    if conf["user"] and conf["bucket_raw"]:
        conf["start"] = upload(conf["user"], conf["folder"], conf["start"], conf["bucket_raw"],
                               conf["profile"])
        with open("conf.json", "w") as f:
            json.dump(conf, f)
    else:
        print("Config conf.json before use upload!")
