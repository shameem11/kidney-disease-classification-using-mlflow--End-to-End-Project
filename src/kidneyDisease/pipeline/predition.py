import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model
        model_path = os.path.join("artifacts", "training", "model.h5")
        model = load_model(model_path)

        # Load and preprocess image
        img = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image to [0,1]

        # Predict
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Define class labels based on the new dataset structure
        class_labels = {0: 'Crystal', 1: 'Normal', 2: 'Stone', 3: 'Tumor'}

        # Get the predicted label
        prediction = class_labels.get(predicted_class, "Unknown")
        
        # Return the result as a dictionary
        return {"image": prediction}

# Example usage:
# pipeline = PredictionPipeline("path_to_image.jpg")
# result = pipeline.predict()
# print(result)
