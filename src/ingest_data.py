import tensorflow
from tensorflow.keras.preprocessing.image import DataImageGenerator
from src.utils import log_time, logging_setup

logger = logging_setup


class ingest_data:
    def __init__(self, img_width, img_height, batch_size) -> None:
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        

    @log_time
    def build_generator(self):
        try:
            logger.info("Start building generator")
            self.train_generator = DataImageGenerator(
                rescale = 1./255,
                rotation_range = 20,
                width_shift_range = 0.2,
                height_shift_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True,
                fill_mode = 'nearest'
            )
            self.val_generator = DataImageGenerator(rescale = 1./255)
            logger.info("Image Generator build successfully")
        except Exception as e:
            logger.error(f"Error while building generator: {str(e)}")
            raise e

    def process_generator(self):
        pass
