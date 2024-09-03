from src.utils import logging_setup, log_time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logger = logging_setup()

class Trainer:
    def __init__(self, data_ingester, model_builder, epochs:int) -> None:
        self.data_ingester = data_ingester
        self.model_builder = model_builder
        self.epochs = epochs
    
    @log_time
    def train(self):
        try:
            logger.info("Start model training")
            steps_per_epoch = len(self.data_ingester.train_dataset)
            val_steps = len(self.data_ingester.val_dataset)
            
            checkpoint = ModelCheckpoint('Checkpoints/model_best.h5', monitor='val_loss', save_best_only=True, verbose=1)
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

            history = self.model_builder.model.fit(
                self.data_ingester.train_dataset,
                epochs=self.epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=self.data_ingester.val_dataset,
                validation_steps=val_steps,
                callbacks=[checkpoint, early_stopping]
            )
        except Exception as e:
            logger.error(f"Error while training model: {str(e)}")
            raise e
