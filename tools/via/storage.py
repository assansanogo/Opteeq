"""
Manage storage between local and aws buckets.
"""
import io
import os
from abc import ABC, abstractmethod

from tools.aws.awsTools import Bucket


class Storage(ABC):
    """
    Storage interface
    """

    @abstractmethod
    def read(self, file) -> bytes:
        """
        Read a file and return it in bytes.
        Need to implement.

        :param file: file
        :return: file bytes
        """
        pass

    @abstractmethod
    def list_files(self) -> list:
        """
        list all file in the storage.

        :return: list of all files.
        """
        pass


class Local(Storage):
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

    def list_files(self) -> list:
        """
        list all file in the storage.

        :return: list of all files
        """
        return os.listdir(self.folder)


class AwsStorage(Bucket, Storage):
    """
    Aws bucket storage
    """

    def __init__(self, bucket_name: str, profile: str = "default"):
        """
        :param bucket_name: name of the bucket.
        :param profile: Choose AWS CLI profile if more than 1 are set up
        """
        super().__init__(bucket_name, profile=profile)


def storage_factory(local: bool, storage_path: str, **kwargs) -> Storage:
    """
    factory return the the good storage object (local or aws) in function of local variable.

    :param local: boolean False use bucket, true local storage
    :param storage_path: path of the image folder or bucket name
    :keyword:
        :key bucket_name (str): name of the bucket.
        :key profile (str): Choose AWS CLI profile if more than 1 are set up.
    :return: Storage object.
    """
    if local:
        return Local(storage_path)
    else:
        return AwsStorage(storage_path, **kwargs)
