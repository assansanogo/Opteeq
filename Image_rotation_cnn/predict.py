# Imports
from keras.preprocessing import image
import numpy as np

def predict_orientation(model: object, image_path: str, threshold: float) -> int:
    """Predicts and returns the orientation of a ticket image.
    Predicts 0 (good orientation) by default except if the model detects another orientation with
    a probability superior to the threshold given as parameter.
    
    :param model: Keras model to be used for the prediction
    :type model: Keras model instance
    :param image_path: path to the image
    :type image_path: str
    :param threshold: threshold between 0-1
    :type threshold: float
    :return: orientation prediction (0, 90, 180 or 270)
    :rtype: int
    """
    img = image.load_img(image_path, target_size=(180, 180))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # Remember that the model was trained on inputs
    # that were preprocessed in the following way:
    img_tensor /= 255.
    
    label_map = {0: 0, 1: 180, 2: 270, 3: 90}
    prediction_poba = model.predict(img_tensor)
    if prediction_poba.max() > threshold:
         prediction = prediction_poba.argmax()
    else:
        prediction = 0

    result = label_map[prediction]
    return result


