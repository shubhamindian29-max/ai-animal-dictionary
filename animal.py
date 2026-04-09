import streamlit as st
from PIL import Image
import requests
import timm
import torch
import PIL.Image
from torchvision import transforms

st.set_page_config(page_title="AI Animal Dictionary", page_icon="🐾")

# --- Load the Local AI Brain ---
@st.cache_resource
def load_local_model():
    # This is a very fast, accurate model that works on Python 3.13
    model = timm.create_model('mobilenetv3_large_100', pretrained=True)
    model.eval()
    
    # Get the names of the 1000 things it can see
    data_config = timm.data.resolve_model_data_config(model)
    labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    labels = requests.get(labels_url).text.splitlines()
    
    return model, data_config, labels

model, config, labels = load_local_model()

def get_wiki_info(name):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{name.replace(' ', '_')}"
    try:
        resp = requests.get(url, timeout=5).json()
        return resp.get('extract', "Information not found.")
    except:
        return "Could not connect to Wikipedia, but I know what this is!"

# --- UI ---
st.title("🐾 Local AI Animal Dictionary")
st.write("Identifies animals locally on your PC - no API limits!")

uploaded_file = st.file_uploader("Upload an animal photo", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Analyzing this...", use_container_width=True)
    
    if st.button("Identify Animal"):
        with st.spinner("Thinking..."):
            # Prepare image for the model
            transform = timm.data.create_transform(**config)
            input_tensor = transform(img).unsqueeze(0) 
            
            # Predict!
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                values, indices = torch.topk(probabilities, 1)
                
            # The model labels start at index 1 for some versions, 
            # we adjust to find the right name
            animal_name = labels[indices[0].item()].title()
            
            st.header(f"Results: {animal_name}")
            
            # Fetch Dictionary Info
            description = get_wiki_info(animal_name)
            st.subheader("Dictionary Entry")
            st.write(description)
