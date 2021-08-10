import boto3
from boto3.dynamodb.conditions import Key
import logging
from botocore.exceptions import ClientError
import os
import json
from typing import Union
from itertools import cycle


class Bucket:
    """
    Aws bucket
    """

    def __init__(self, bucket_name: str):
        """
        :param bucket_name: bucket name
        """
        self.client = boto3.resource("s3").Bucket(bucket_name)

    def upload(self, file_path: str, object_name: str = None) -> None:
        """
        Upload a file to Bucket

        :param file_path: path of the file to upload
        :param object_name: S3 object name. If not specified then file_name is used
        """
        if object_name is None:
            object_name = os.path.basename(file_path)
        try:
            self.client.upload_file(file_path, object_name)
        except ClientError as e:
            logging.error(e)

    def put_object(self, content: Union[str, dict, int, float], key: str) -> None:
        """
        Function to upload python variable in s3.

        Convert dict in json and other python variable in string, convert in bytes and upload to s3.

        :param content: python variable to upload
        :param key: key for s3 object
        """
        if isinstance(content, dict):
            content = json.dumps(content)
        else:
            content = str(content)
        self.client.put_object(Body=bytes(content.encode("UTF-8")), Key=key)

    def download(self, key: str, directory: str) -> None:
        """
        Download a file from Bucket

        :param key: object key
        :param file_path: destination path directory
        """
        self.client.download_file(key, os.path.join(directory, key))

    def read(self, key: str) -> bytes:
        """
        load in memory object from s3.

        :param key: object key
        :return: object bytes
        """
        return self.client.Object(key).get()["Body"].read()

    def get_last_put(self) -> Union[str, None]:
        """
        get the key of the last put object

        :return: key of the last put object, return None if the bucket is empty
        """
        sorted_files = sorted(self.client.objects.filter(), key=lambda x: x.last_modified,
                              reverse=True)
        if sorted_files:
            return sorted_files[0].key
        else:
            return None

    def list_object_search_key(self, key_part: str) -> list:
        """
        list all object in the bucket which contains key_part in their key
        :param key_part: string part of the key
        :return: list of object
        """

        return [i.key for i in self.client.objects.all() if key_part in i.key]


class BucketCounter(Bucket):
    """
    Aws bucket with a cycle on annotator names.
    """

    def __init__(self, bucket_name: str, annotator_names: list):
        """
        :param bucket_name: bucket name
        :param annotator_names: list the name of all the annotator
        """
        super().__init__(bucket_name)
        self.annotator_cycle = cycle((lambda i: annotator_names[i:] + annotator_names[:i])(
            self.get_last_uploader_index(annotator_names) + 1))

    def get_last_uploader_index(self, annotator_names: list) -> int:
        """
        Get the index of the last uploader in the annotator list name
        :param annotator_names: list the name of all the annotator
        :return: index of last uploader, -1 if error
        """
        try:
            return annotator_names.index(self.get_last_put().split("_")[0])
        # value error not in list, attribute error try to split None
        except (ValueError, AttributeError):
            # -1: the next item will be the first
            return -1

    def put_object_annotator(self, content: Union[str, dict, int, float], key: str) -> str:
        """
        Function to upload python variable in s3.

        Convert dict in json and other python variable in string, convert in bytes and upload to s3.

        Add cycle annotator on the key.

        :param content: python variable to upload
        :param key: key for s3 object
        :return: name of the annotator for the curent cycle
        """
        annotator = next(self.annotator_cycle)
        self.put_object(content, f"{annotator}_{key}")
        return annotator


class DynamoDB:
    """
    Aws dynamoDB
    """

    def __init__(self, region, table_name):
        self.client = boto3.resource('dynamodb', region_name=region).Table(table_name)

    def query(self, **kwargs) -> dict:
        """
        Dynamodb query.

        :param kwargs: all possible kwargs for query cf dynamo db doc.
        :return: query result
        """
        return self.client.query(**kwargs)

    def query_index_eq(self, index: str, key_name: str, value: str) -> dict:
        """
        Query on a secondary index equal value

        :param index: index name
        :param key_name: key name
        :param value: key value
        :return: query result
        """
        return self.query(IndexName=index, KeyConditionExpression=Key(key_name).eq(value))

    def get_keys_annotator(self, annotator: str) -> list:
        """
        Query on the annotator index and get all the key of standardised image.

        :param annotator: annotator name value
        :return: list all key of standardised image with the given annotator.
        """
        items = self.query_index_eq("annotator-index", "annotator", annotator)["Items"]
        return [i["standardised"] for i in items]

    def update(self, **kwargs) -> Union[dict, None, Exception]:
        """
        DynamoDB update

        :param kwargs: all possible kwargs dynamoDB update cf docs
        :return: response or exception
        """
        try:
            response = self.client.update_item(**kwargs)
        except ClientError as e:
            if e.response['Error']['Code'] == "ConditionalCheckFailedException":
                print(e.response['Error']['Message'])
            else:
                raise
        else:
            return response

    def update_annotator(self, key: str, annotator: str) -> None:
        """
        Update the annotator value of the given key.

        :param key: primary key value
        :param annotator: new annotator name
        """
        return self.update(Key={'standardised': key},
                           UpdateExpression="set annotator=:a",
                           ExpressionAttributeValues={
                               ':a': annotator,
                           },
                           ReturnValues="UPDATED_NEW")


class Rekognition:
    def __init__(self, bucket=None, region="eu-west-1"):
        """
        :param bucket: bucket name
        :param region: region of the aws rekognition service
        """
        self.bucket = bucket
        self.client = boto3.client('rekognition', region)

    def get_text(self, key: str) -> dict:
        """
        Get text annotation for an image.

        :param key: key of image in the bucket
        :return: dict response from aws rekognition
        """
        return self.client.detect_text(Image={'S3Object': {'Bucket': self.bucket, 'Name': key}})
