import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from src.utils import logging_setup

logger = logging_setup()

class ModelBuilder:
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self):
        try:
            logger.info("Start building model")
            self.model = Sequential()
            self.model.add(Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
            self.model.add(MaxPooling2D((2,2)))
            self.model.add(Conv2D(64, (3,3), activation='relu'))
            self.model.add(MaxPooling2D((2,2)))
            self.model.add(Conv2D(128, (3,3), activation='relu'))
            self.model.add(MaxPooling2D((2,2)))
            self.model.add(Conv2D(128, (3,3), activation='relu'))
            self.model.add(MaxPooling2D((2,2)))
            self.model.add(Flatten())
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dense(self.num_classes, activation='sigmoid'))

            logger.info("Model build successfully")

        except Exception as e:
            logger.error(f"Error while building model")
            raise e
        
    def compile_model(self):
        try:
            logger.info("start compiling model")
            self.model.compile(loss='binary_crossentropy',
                               optimizer='adam',
                               metrics=['accuracy'])
            logger.info("Model compiled successfully")
        except Exception as e:
            logger.error(f"Error in compiling model")
            raise e
                      