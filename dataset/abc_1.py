import streamlit as st
from PIL import Image
import requests

# Set up the Streamlit app
st.title('Image Captioning Application')
st.write('Upload an image, and the model will generate a caption for it.')

# Upload the image
uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Perform image captioning
    if st.button('Generate Caption'):
        st.write('Generating caption...')
        
        # Convert image to bytes
        image_bytes = uploaded_image.read()
        
        # Make a POST request to the Flask backend
        response = requests.post('http://127.0.0.1:5000/generate_caption', files={'image': image_bytes})
        
        if response.status_code == 200:
            caption = response.json().get('caption', 'No caption generated')
            st.write(f'Caption: {caption}')
        else:
            st.write('Error generating caption:', response.json().get('error', 'Unknown error'))

