import os
from pathlib import Path
import time


def set_project_root() -> Path:
    return Path(__file__).parent.parent


def build_directory(path, directory) -> str:
    dir_path = os.path.join(path, directory)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path


def create_project_structure(model_name, dataset_name):
    # Set working directory to project root
    set_project_root()

    # Create timestamp for training job
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    experiment_name = model_name + "-" + timestamp

    # Data path
    base_data_path = build_directory(os.getcwd(), 'Data')
    print("Data path: {}".format(base_data_path))

    # Path to dataset
    dataset_path = build_directory(base_data_path, dataset_name)
    print("Dataset path: {}".format(dataset_path))

    base_experiments_path = build_directory(os.getcwd(), 'Experiments')
    print("Base experiments path: {}".format(base_experiments_path))

    model_path = build_directory(base_experiments_path, model_name)

    training_job_path = build_directory(model_path, experiment_name)
    print("Training job path: {}".format(training_job_path))

    log_path = build_directory(training_job_path, "logs-" + timestamp)
    print("Log directory path: {}".format(log_path))
    return base_data_path, dataset_path, training_job_path, log_path