from kidneyDisease.config.configuration import ConfigurationManager
from kidneyDisease.components.prepare_base_model import PrepareBaseModel
from kidneyDisease import logger
import tensorflow as tf

STAGE_NAME = 'Prepare base model'


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        # Initialize the PrepareBaseModel class with config
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        # Get the base model (VGG16 or whichever base is being used)
        prepare_base_model.get_base_model()
        # Unfreezing last two layers (you can make this configurable)
        unfreeze_layers = prepare_base_model_config.unfreeze_layers if hasattr(prepare_base_model_config, 'unfreeze_layers') else [-1, -2]
        # Updating the base model with unfreezed layers
        prepare_base_model.update_base_model(unfreeze_layers=unfreeze_layers)
        # Log the model summary after unfreezing layers for tracking
        logger.info(f"Model summary after unfreezing layers: \n{prepare_base_model.full_model.summary()}")

        # Adding early stopping to prevent overfitting (you can customize the patience)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', save_best_only=True)
        ]
        
        # Optionally, you can log additional metrics like learning rate schedule or training performance
    
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")        
        # Run the pipeline
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    
    except Exception as e:
        logger.exception(e)
        raise e
