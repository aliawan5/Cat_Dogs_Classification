import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.utils import log_time, logging_setup

logger = logging_setup()


class IngestData:
    def __init__(self, img_width, img_height, batch_size, data_path) -> None:
        self.data_path = data_path
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        

    @log_time
    def build_generator(self):
        try:
            logger.info("Start building generator")
            self.train_generator = ImageDataGenerator(
                rescale = 1./255,
                rotation_range = 20,
                width_shift_range = 0.2,
                height_shift_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True,
                fill_mode = 'nearest',
                validation_split = 0.2
            )
            self.val_generator = ImageDataGenerator(rescale = 1./255)
            logger.info("Image Generator build successfully")
        except Exception as e:
            logger.error(f"Error while building generator: {str(e)}")
            raise e

    def process_generator(self):
        try:
            logger.info("Start Processing Generator")

            self.train_dataset = self.train_generator.flow_from_directory(
                self.data_path,
                target_size = (self.img_width, self.img_height),
                batch_size = self.batch_size,
                class_mode = 'binary',
                subset = 'training'
                )
            
            self.val_dataset = self.val_generator.flow_from_directory(
                self.data_path,
                target_size = (self.img_width, self.img_height),
                batch_size = self.batch_size,
                class_mode = 'binary',
                subset = 'validation'
            )
            # sample_train = next(self.train_dataset)
            # logger.info(f"Sample batch from train_dataset: {sample_train}")

            # sample_val = next(self.val_dataset)
            # logger.info(f"Sample batch from train_dataset: {sample_val}")

            logger.info("Generator processed successfully")
        except Exception as e:
            logger.error(f"Error in Processing generator: {str(e)}")
            raise e
