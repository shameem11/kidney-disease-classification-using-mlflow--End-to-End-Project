
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
    def _prepare_full_model(model, classes, learning_rate, unfreeze_layers=None):
        # If specific layers need to be unfrozen, unfreeze those
        if unfreeze_layers is not None:
            for layer in unfreeze_layers:
                if 0 <= layer < len(model.layers):
                    model.layers[layer].trainable = True

        # Flatten the output of the base model
        flatten_in = tf.keras.layers.Flatten()(model.output)

        # Add a Batch Normalization layer to stabilize and accelerate training
        batch_norm = tf.keras.layers.BatchNormalization()(flatten_in)

        # Add dropout layer to reduce overfitting
        dropout = tf.keras.layers.Dropout(0.5)(batch_norm)

        # Add the final dense layer with softmax activation and L2 regularization
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)  # L2 Regularization
        )(dropout)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Learning Rate Scheduling for better convergence
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=10000,
            decay_rate=0.96,
            staircase=True
        )

        # Compile the model with Adam optimizer and a learning rate scheduler
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
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
            learning_rate=self.config.params_learning_rate,
            unfreeze_layers=unfreeze_layers  # Pass layers to unfreeze for fine-tuning
        )

        # Save the updated model
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train_model(self, train_data, val_data, epochs):
        # Adding EarlyStopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Can also use 'val_accuracy' depending on preference
            patience=3,  # Stop if no improvement after 3 epochs
            restore_best_weights=True,  # Restores weights from the best performing epoch
            verbose=1
        )

        # Train the model
        self.full_model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=[early_stopping]  # Include the EarlyStopping callback
        )
