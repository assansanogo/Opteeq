import os
import time
from math import ceil

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing import load_img
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from tools import build_directory


def load_model(model_path):
    files = os.listdir(model_path)

    for file in files:
        if file.endswith(".h5"):
            model = load_model(os.path.join(model_path, file))
    return model


def plot_model_history(model_history, eval_path, acc='accuracy', val_acc='val_accuracy'):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(model_history.history[acc]) + 1), model_history.history[acc])
    axs[0].plot(range(1, len(model_history.history[val_acc]) + 1), model_history.history[val_acc])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history[acc]) + 1), len(model_history.history[acc]) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig(os.path.join(eval_path, "model_history.png"))
    plt.show()
    return


def plot_confusion_matrix(y_test, y_final, eval_path):
    # compute the confusion matrix
    confusion_matrix = confusion_matrix(y_test, y_final)
    # plot the confusion matrix
    f, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion_matrix, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f', ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(eval_path, "confusion_matrix.png"))
    plt.show()
    return


def classification_report(y_test, y_final, eval_path):
    # Generate a classification report
    report = classification_report(y_test, y_final, target_names=['0', '90', '180', '270'])

    # Generate a classification report
    report_dict = classification_report(y_test, y_final, target_names=['0', '90', '180', '270'], output_dict=True)
    report_df = pd.DataFrame.from_dict(report_dict)
    report_df.to_csv(os.path.join(eval_path, "classification_report.csv"))
    return


def plot_sample_predictions(test_df, training_job_path):
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    sample_preds_path = build_directory(training_job_path, "sample_predictions")

    sample_test = test_df.sample(n=9).reset_index()

    fig = plt.figure(figsize=(12, 12))

    for i, row in sample_test.iterrows():
        i += 1

        col = 'red' if row['pred'] == row['category'] else 'green'

        img = load_img(os.path.join(test_data_path, row['filename']), target_size=(256, 256))

        ax = fig.add_subplot(3, 3, i)
        ax.xaxis.label.set_color(col)
        ax.imshow(img)
        ax.set_xlabel("Category: {}, Prediction: {}".format(int(row['category']), row['pred']))
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                       labelleft=False)

    plt.tight_layout()
    plt.savefig(os.path.join(sample_preds_path, "example_preds_{}.png".format(timestamp)))
    plt.show()
    return


def plot_sample_errors(test_df, training_job_path, test_data_path):
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    sample_preds_path = build_directory(training_job_path, "sample_errors")

    sample_test = test_df.sample(n=9).reset_index()
    errors = {}

    for i, row in test_df.iterrows():
        if int(row['category']) != row['pred']:
            errors.update({i: row})
        else:
            pass

    errors_df = pd.DataFrame.from_dict(errors).transpose().reset_index(drop=True)

    fig = plt.figure(figsize=(12,12))

    for i, row in errors_df[:9].iterrows():
        i+=1

        col = 'green' if row['pred'] == row['category'] else 'red'

        img = load_img(os.path.join(test_data_path, row['filename']), target_size=(256, 256))

        ax = fig.add_subplot(3, 3, i)
        ax.xaxis.label.set_color(col)
        ax.imshow(img)
        ax.set_xlabel("Category: {}, Prediction: {}".format(int(row['category']), row['pred']))
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

    plt.tight_layout()
    plt.savefig(os.path.join(sample_preds_path, "example_errors_{}.png".format(timestamp)))
    plt.show()
    return


def model_eval(history, test_df, test_generator, batch_size, training_job_path, dataset_path):
    # Build evaluation path
    eval_path = build_directory(training_job_path, "evaluation_report")

    # Test data path
    test_data_path = os.path.join(dataset_path, "test")

    # Plot model history
    plot_model_history(history, eval_path)

    # Load in best model
    model = load_model(training_job_path)

    # Evaluate model using test set
    test_loss, test_acc = model.evaluate_generator(test_generator, steps=ceil(len(test_df) / batch_size))
    print('Overall test accuracy:', test_acc)

    # Get predictions using predict_generator
    y_test = test_df["category"].astype(int)
    y_pred = model.predict_generator(test_generator)

    # Set threshold for prediction
    threshold = 0.5

    # If prediction probability for class if greater than threshold value to 1 else 0
    y_final = np.where(y_pred > threshold, 1, 0).argmax(axis=1)
    test_df['pred'] = y_final

    # Build confusion matrix
    plot_confusion_matrix(y_test, y_final, eval_path)

    # Build evaluation report
    classification_report(y_test, y_final, eval_path)

    # Get sample predictions
    plot_sample_predictions(test_df, training_job_path, test_data_path)

    # Get sample errors
    plot_sample_errors(test_df, training_job_path, test_data_path)
    return
