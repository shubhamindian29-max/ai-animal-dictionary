import streamlit as st
from PIL import Image
import requests

st.set_page_config(page_title="AI Animal Dictionary", page_icon="🐾")

st.title("🐾 AI Animal Dictionary")
st.write("Upload a photo of an animal to learn about it!")

# Use a free API for identification to avoid TensorFlow version issues
API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
headers = {"Authorization": "Bearer hf_VvGToBvXmYpYpYpYpYpYpYp"} # This is a placeholder

def query(img_bytes):
    response = requests.post(API_URL, data=img_bytes)
    return response.json()

def get_wiki_info(name):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{name.replace(' ', '_')}"
    resp = requests.get(url).json()
    return resp.get('extract', "Information not found.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    
    if st.button("Identify Animal"):
        with st.spinner("Analyzing..."):
            # Prepare image for API
            uploaded_file.seek(0)
            img_bytes = uploaded_file.read()
            
            # Get Prediction
            results = query(img_bytes)
            if results and 'label' in results[0]:
                animal_name = results[0]['label'].split(',')[0].title()
                st.header(f"It's a {animal_name}!")
                
                # Get Dictionary Info
                info = get_wiki_info(animal_name)
                st.write(info)
            else:
                st.error("Could not identify. Try a clearer photo!")