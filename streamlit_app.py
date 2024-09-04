import streamlit as st
from PIL import Image
import tensorflow as tf
from transformers import pipeline

# Load the trained model
model = tf.keras.models.load_model('coco_image_recognition_model.h5')

# Load the conversational model
nlp_model = pipeline('conversational', model="microsoft/DialoGPT-medium")

# Define label map for COCO dataset
from data_preparation import ds_info
label_map = ds_info.features['objects']['label'].int2str

# Streamlit UI
st.title("Conversational Image Recognition Chatbot")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    # Predict using the model
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions[0]).numpy()
    predicted_label = label_map(predicted_class)

    st.write(f"Predicted Object: {predicted_label}")

    # Conversational response
    user_query = st.text_input("Ask something about the image:")
    if user_query:
        chat_history = nlp_model(f"Image contains a {predicted_label}. {user_query}")
        response = chat_history[-1]['generated_text']
        st.write(f"Chatbot: {response}")
