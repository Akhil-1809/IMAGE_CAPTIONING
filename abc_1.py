import streamlit as st
import requests
from PIL import Image
import io

# Set the URL of your Flask backend
FLASK_BACKEND_URL = 'http://127.0.0.1:5000/generate_caption'

# Streamlit app title
st.title("Image Captioning Application")

# Description
st.write("Upload an image, and the model will generate a caption for it.")

# File uploader for image
uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    try:
        # Open the uploaded image file
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Convert image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()

        # Button to generate caption
        if st.button('Generate Caption'):
            # Show a loading message
            with st.spinner('Generating caption...'):
                # Send the image to the Flask backend
                response = requests.post(FLASK_BACKEND_URL, files={'image': ('image.jpg', image_bytes, 'image/jpeg')})

                # Handle the response from the backend
                if response.status_code == 200:
                    caption = response.json().get('caption', 'No caption generated')
                    st.write(f'**Caption:** {caption}')
                else:
                    st.write(f"Error: {response.json().get('error', 'Unknown error')}")

    except Exception as e:
        st.error(f"Error loading the image: {e}")
