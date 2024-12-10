import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load your trained models
pathmnist_model = load_model("model-1_path.h5")
pneumoniamist_model = load_model("model_pneumonia_1.h5")

# Define class labels for PathMNIST and PneumoniaMist
pathmnist_class_labels = ['adipose', 'background', 'debris', 'lymphocytes', 'mucus', 'smooth muscle', 'normal colon mucosa', 'cancer-associated stroma', 'colorectal adenocarcinoma epithelium']
pneumoniamist_class_labels = ['Normal','Pneumonia']

# Function to preprocess the image
def preprocess_image(image, model_name):
    # Resize the image to match the input shape of your model
    if model_name == "PathMNIST" or model_name == "PneumoniaMist":
        image = image.resize((32, 32))
    # Convert the image to array
    image = np.array(image)
    # Normalize the image
    image = image / 255.0
    # Expand the dimensions to match the input shape of your model
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
def main():
    st.title("Image Classification")
    st.write("Upload an image for classification")

    # Select model
    model_name = st.selectbox("Select Model", ["PathMNIST", "PneumoniaMist"])

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image, model_name)

        # Make predictions based on selected model
        if model_name == "PathMNIST":
            model = pathmnist_model
            class_labels = pathmnist_class_labels
        elif model_name == "PneumoniaMist":
            model = pneumoniamist_model
            class_labels = pneumoniamist_class_labels

        # Make predictions
        prediction = model.predict(processed_image)
        # Get the predicted class label
        predicted_class = np.argmax(prediction[0])
        predicted_label = class_labels[predicted_class]

        # Display prediction
        st.write("Predicted Class:", predicted_label)
        st.write("Confidence:", prediction[0][predicted_class])

if __name__ == "__main__":
    main()
