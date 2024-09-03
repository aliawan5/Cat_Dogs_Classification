from src.ingest_data import IngestData
from src.build_model import ModelBuilder
from src.train_model import Trainer
from src.evaluate_model import Evaluate


def main():
    img_height = 150
    img_width = 150
    batch_size = 32
    num_classes = 1
    data_path = r'C:\Users\FINE\Desktop\animal'
    model_path = 'Checkpoint/model_best.keras'
    img_path = r'C:\Users\FINE\Desktop\Dog_Cat_Classification\data\pic.jpg'
    epochs = 15


    data_ingester = IngestData(img_width, img_height, batch_size, data_path)
    data_ingester.build_generator()
    data_ingester.process_generator()

    model_builder = ModelBuilder(num_classes)
    model_builder.build_model()
    model_builder.compile_model()

    trainer = Trainer(data_ingester, model_builder, epochs)
    trainer.train()

    Evaluation = Evaluate(model_path, img_path, img_height, img_width)
    Evaluation.evaluate()


if __name__ == "__main__":
    main()