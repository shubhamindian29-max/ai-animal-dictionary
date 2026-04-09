import streamlit as st
from PIL import Image
import requests
import timm
import torch
import certifi
import os

# --- 1. SECURITY & CONNECTION CONFIG ---
# This fixes the "SSL" and "Connection Error" issues on Windows
os.environ['SSL_CERT_FILE'] = certifi.where()

st.set_page_config(
    page_title="Ultimate Animal Dictionary", 
    page_icon="🐾",
    layout="centered"
)

# --- 2. THE AI MODEL (THE BRAIN) ---
@st.cache_resource
def load_model():
    # ResNet50 is very accurate for identifying animals
    model = timm.create_model('resnet50', pretrained=True)
    model.eval()
    config = timm.data.resolve_model_data_config(model)
    
    # Load the labels (the names of the 1000 things the AI knows)
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        labels = requests.get(labels_url).text.splitlines()
    except:
        # Fallback if the URL is down
        labels = ["Animal"] * 1000 
    
    return model, config, labels

model, config, labels = load_model()

# --- 3. THE DICTIONARY (WIKIPEDIA API) ---
def get_wiki_info(name):
    # Wikipedia needs to think you are a real browser, not a bot
    headers = {
        'User-Agent': 'AnimalDictionaryApp/1.0 (https://yourwebsite.com; your-email@example.com)'
    }
    
    # Format the name for the URL (e.g., "Golden Retriever" -> "Golden_Retriever")
    search_name = name.replace(' ', '_')
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{search_name}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('extract', "No summary found.")
        else:
            return f"I found the animal ({name}), but Wikipedia doesn't have a summary for this specific category."
            
    except Exception as e:
        return f"Could not connect to the Encyclopedia. Error: {e}"

# --- 4. THE WEBSITE INTERFACE ---
st.title("🐾 AI Animal Dictionary")
st.markdown("---")

uploaded_file = st.file_uploader("Step 1: Upload a clear photo of an animal", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Open and show the image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Target Animal", use_container_width=True)
    
    if st.button("Step 2: Identify and Search Dictionary"):
        with st.spinner("Analyzing species and fetching data..."):
            
            # Prepare image for AI
            transform = timm.data.create_transform(**config)
            input_tensor = transform(img).unsqueeze(0) 
            
            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                values, indices = torch.topk(probabilities, 1)
            
            # Get the clean name
            prediction = labels[indices[0].item()].strip().title()
            
            # Display Result
            st.success(f"**Identity Confirmed:** {prediction}")
            
            # Display Dictionary Info
            st.markdown("### 📖 Dictionary Entry")
            description = get_wiki_info(prediction)
            st.write(description)
            
            st.info("Tip: If the identity is slightly off, try a photo with better lighting or a different angle!")

# --- 5. FOOTER ---
st.markdown("---")
st.caption("Powered by ResNet50 Vision Model and Wikipedia API")
