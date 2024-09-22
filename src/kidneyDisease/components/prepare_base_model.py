
import os 
import tensorflow as tf 
from pathlib import Path
from kidneyDisease.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig) -> None:
        self.config = config

    def get_base_model(self):
        # Load base VGG16 model
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        # Save the base model
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate, unfreeze_layers=None):
        # If specific layers need to be unfrozen, unfreeze those
        if unfreeze_layers is not None:
            for layer in unfreeze_layers:
                if 0 <= layer < len(model.layers):
                    model.layers[layer].trainable = True
        
        # Add new layers to the model
        flatten_in = tf.keras.layers.Flatten()(model.output)
        
        # Add dropout layer to reduce overfitting
        dropout = tf.keras.layers.Dropout(0.5)(flatten_in)
        
        # Add the final dense layer with softmax activation
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)  # L2 Regularization
        )(dropout)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Using Adam optimizer for better performance
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self, unfreeze_layers=None):
        # Prepare full model with additional layers and fine-tuning
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,  # Modify freeze_all as necessary
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
            unfreeze_layers=unfreeze_layers  # Pass layers to unfreeze for fine-tuning
        )

        # Save the updated model
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
