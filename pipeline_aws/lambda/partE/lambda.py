import json
import os
import csv
import boto3
import dateutil.parser as dp


def lambda_handler(event, context):
    # Extract needed information from event
    bucket_name = event["Records"][0]["s3"]["bucket"]["name"]
    object_key = event["Records"][0]["s3"]["object"]["key"]
    timestamp_iso8601 = event["Records"][0]["eventTime"]

    # Convert iso8601 timestamp to Unix epoch timestamp
    timestamp_iso8601_parsed = dp.parse(timestamp_iso8601)
    timestamp = round(timestamp_iso8601_parsed.timestamp())

    # Get Dynamodb table name from enviroment variable
    table_name = os.environ['dynamodb_table_name']

    # Access DynamoDB table and S3 bucket
    table = boto3.resource('dynamodb', region_name='eu-west-1').Table(table_name)
    bucket = boto3.resource('s3').Bucket(bucket_name)

    # Download file in temp folder
    local_file_name = '/tmp/' + object_key
    bucket.download_file(object_key, local_file_name)

    # Extract all standard keys in the csv
    stand_keys = set()
    with open(local_file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[0].endswith('.jpg'):
                stand_keys.add(row[0])
    stand_keys = list(stand_keys)

    # Update DynamoDB table for all standard keys found in the annotations csv
    for stand_Key in stand_keys:
        response = table.update_item(
            Key={'standKey': stand_Key},
            UpdateExpression="set finalAnotTime=:t, finalAnotKey=:k",
            ExpressionAttributeValues={
                ':t': timestamp,
                ':k': object_key
            },
            ReturnValues="UPDATED_NEW"
        )

    return {
        'statusCode': 200
    }
