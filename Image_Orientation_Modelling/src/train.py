import tensorflow as tf
from tools import create_project_structure
from dataset import build_dataset
from model import build_model_vgg16
from eval import model_eval

dataset_name = "Dataset_v1"
model_name = "VGG16"

FAST_RUN = False
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
batch_size = 16
epochs = 20

image_size = IMAGE_SIZE
input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def train():
    # Setup project root and build directories for experiment
    dataset_path, training_job_path, log_path = create_project_structure(model_name, dataset_name)

    # Load in data and build dataset
    train_df, train_generator, test_df, test_generator, val_df, val_generator = build_dataset(dataset_path)

    # Build pretrained model
    model, tensorboard_callback, es, mc = build_model_vgg16()

    total_train = len(train_df)
    total_val = len(val_df)

    # fine-tune the model
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=total_val // batch_size,
        steps_per_epoch=total_train // batch_size,
        callbacks=[tensorboard_callback, es, mc])

    model_eval(history, test_df, test_generator, batch_size, training_job_path, dataset_path)


if __name__=="__main__":
    if tf.test.gpu_device_name():
        print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
        print("Running model training")
        train()
    else:
        print("Please install GPU version of TF")