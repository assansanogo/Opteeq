"""
Part C cf schema
"""
from tools.aws.awsTools import DynamoDB, BucketCounter
from tools.via.via_converter import via_json
import time
import json
from tqdm import tqdm


def main(region: str, table_name: str, batch_size: int, bucket_in: str, bucket_out: str,
         annotator_names: list, profile: str = "default") -> None:
    """
    Get all image without annotator from dynamoDB. Generate json via with annotation by batch and
    upload to s3.
    update dynamoDB with annotator name.

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
    if len(key_list := table.get_keys_annotator("0")) > batch_size:
        for i in tqdm(range(len(key_list) // batch_size), desc="batch", leave=False):
            batch = key_list[i * batch_size: (i + 1) * batch_size]
            json = via_json(batch, bucket_in, profile=profile)
            file_name = f'{int(time.time())}_{i}.json'
            json['_via_settings']["project"]["name"] = file_name.split(".")[0]
            annotator = bucket.put_object_annotator(json, file_name)
            # todo find if it is possible to update all in one
            for key in batch:
                table.update_annotator(key, annotator)


if __name__ == '__main__':
    with(open("conf.json", "r")) as f:
        conf = json.load(f)
    if conf["dynamoDB"]["region"] and conf["dynamoDB"]["table"] and conf["bucket_standardized"] and \
            conf["bucket_initial_annotation"]:
        main(conf["dynamoDB"]["region"], conf["dynamoDB"]["table"], conf["batch_size"],
             conf["bucket_standardized"], conf["bucket_initial_annotation"], conf["annotator_list"],
             conf["profile"])
    else:
        print("edit config and add missing argument")
