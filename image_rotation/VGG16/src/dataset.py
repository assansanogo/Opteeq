import os
import zipfile

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator


def read_data(in_dir, data_aug=False):
    categories = []
    filenames = []

    if data_aug == False:
        files = [file for file in os.listdir(in_dir) if file[-10:-7] not in ['_b_', '_n_']]
    else:
        files = os.listdir(in_dir)

    for filename in files:
        filenames.append(filename)
        if '_000' in filename:
            categories.append('0')
        elif '_090' in filename:
            categories.append('1')
        elif '_180' in filename:
            categories.append('2')
        elif '_270' in filename:
            categories.append('3')

    # DataFrame with all files and their class
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    return df

def unzip_data(dataset_path):
    dirs = os.listdir(dataset_path)

    for d in dirs:

        if d.endswith(".zip"):
            print("Found zip file: {}".format(d))

            target_dir = d.split(".")[0]  # Get target folder name
            path_to_zip_file = os.path.join(dataset_path, d)
            directory_to_extract_to = os.path.join(dataset_path, target_dir)

            try:
                with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                    zip_ref.extractall(directory_to_extract_to)
                print("{} successfully extracted contents to {}".format(d, directory_to_extract_to))

                try:
                    print("Removing zip file {}...".format(d))
                    os.remove(path_to_zip_file)
                    print("{} successfully removed.".format(d))
                except:
                    print("Error, {} could not be removed".format(d))

            except:
                print("Error, unable to extract files from file: {}".format(d))

def build_dataset(image_size, batch_size, dataset_path):
    # Unzip data
    unzip_data(dataset_path)

    # Set paths to folders container train, val and test data
    train_data_path = os.path.join(dataset_path, "train_224")
    val_data_path = os.path.join(dataset_path, "val_224")
    test_data_path = os.path.join(dataset_path, "test_224")

    # Read in data
    train_df = read_data(train_data_path, data_aug=True)
    val_df = read_data(val_data_path, data_aug=True)
    test_df = read_data(test_data_path)

    # Build generator for training images
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=False,
        vertical_flip=False,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        train_data_path,
        x_col='filename',
        y_col='category',
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size
    )

    # Build generator for validation images
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        val_data_path,
        x_col='filename',
        y_col='category',
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size
    )

    # Build generator for test images
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        test_data_path,
        x_col='filename',
        y_col='category',
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )
