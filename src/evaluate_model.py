import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from src.utils import logging_setup

logger = logging_setup()

class Evaluate:
    def __init__(self, model_path, img_path, img_height, img_width) -> None:
        self.model_path = model_path
        self.img_path = img_path
        self.img_height = img_height
        self.img_width = img_width

    def evaluate(self):
        try:
            logger.info("Loading pretrained model")
            self.model = tf.keras.models.load_model(self.model_path)

            self.img = image.load_img(self.img_path, target_size=(self.img_height, self.img_width))
            self.image_array = image.img_to_array(self.img)
            self.image_array = np.expand_dims(self.image_array, 0)

            prediction = self.model.predict(self.image_array)
            if self.num_classes == 1:
                self.prediction_class = 1 if prediction[0][0] > 0.5 else 0
            else:
                self.prediction_class = np.argmax(prediction[0])

            logger.info("Image successfully predicted")
        except Exception as e:
            logger.error(f"Error while evaluating model: {str(e)}")
            raise e
