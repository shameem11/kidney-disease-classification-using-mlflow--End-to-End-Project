import os
import streamlit as st
from kidneyDisease.pipeline.predition import PredictionPipeline
from kidneyDisease.utils.common import decodeImage, encodeImageIntoBase64

# Streamlit app
def main():
    st.title("Kidney Disease Classification")

    # Upload an image
    uploaded_image = st.file_uploader("Choose a CT Scan image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Save the uploaded image to a temporary path
        temp_file_path = os.path.join("temp", uploaded_image.name)
        if not os.path.exists("temp"):
            os.makedirs("temp")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_image.read())

        try:
            # Encode the image into Base64 using the function from utils.common
            encoded_image = encodeImageIntoBase64(temp_file_path)

            # Decode the image back (if needed)
            decoded_file_path = os.path.join("temp", "decoded_" + uploaded_image.name)
            decodeImage(encoded_image, decoded_file_path)

            # Perform prediction
            prediction_pipeline = PredictionPipeline(temp_file_path)
            prediction_result = prediction_pipeline.predict()

            # Display the results
            st.write(f"Prediction: {prediction_result}")

            # Optionally display the decoded image
            st.image(decoded_file_path, caption='Decoded Image', use_column_width=True)

        except FileNotFoundError as e:
            st.error(str(e))

        finally:
            # Clean up temporary files
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(decoded_file_path):
                os.remove(decoded_file_path)

if __name__ == "__main__":
    main()
