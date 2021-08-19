"""
Part C cf schema
"""
import json
import time

from tqdm import tqdm

from tools.aws.awsTools import DynamoDB, BucketCounter
from tools.via.via_converter import ViaConverter


def run_generation(cloud_provider, region: str, table_name: str, batch_size: int, bucket_in: str,
                   bucket_out: str,
                   annotator_names: list, profile: str = "default") -> None:
    """
    Get all image without annotator from dynamoDB. Generate json via with annotation by batch and
    upload to s3.
    update dynamoDB with annotator name.

    :param cloud_provider: cloud provider name "aws" for aws rekognition, "gcloud for gcloud vison
    :param region: dynamoDB table region
    :param table_name: dynamoDB table name
    :param batch_size: batch size for via json
    :param bucket_in: bucket with standardised image
    :param bucket_out: bucket where via json are upload
    :param annotator_names: list the name of all the annotator
    :param profile: Choose AWS CLI profile if more than 1 are set up
    """
    table = DynamoDB(region, table_name, profile)
    bucket = BucketCounter(bucket_out, annotator_names, profile)
    via = ViaConverter(cloud_provider, False, bucket_in, profile)
    if len(key_list := table.get_keys_annotator("0")) > batch_size:
        for i in tqdm(range(len(key_list) // batch_size), desc="batch", leave=False):
            batch = key_list[i * batch_size: (i + 1) * batch_size]
            json_annotations = via.via_json(batch)
            file_name = f'{int(time.time())}_{i}.json'
            json_annotations['_via_settings']["project"]["name"] = file_name.split(".")[0]
            annotator = bucket.put_object_annotator(json_annotations, file_name)
            # todo find if it is possible to update all in one
            for key in batch:
                table.update_annotator(key, annotator, file_name)


if __name__ == '__main__':
    with(open("conf.json", "r")) as f:
        conf = json.load(f)
    if conf["dynamoDB"]["region"] and conf["dynamoDB"]["table"] and \
            conf["bucket_standardized"] and conf["bucket_initial_annotation"]:
        run_generation(conf["cloud_provider"], conf["dynamoDB"]["region"],
                       conf["dynamoDB"]["table"],
                       conf["batch_size"],
                       conf["bucket_standardized"], conf["bucket_initial_annotation"],
                       conf["annotator_list"],
                       conf["profile"])
    else:
        print("edit config and add missing argument")
