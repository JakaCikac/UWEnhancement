import streamlit as st
from pathlib import Path
from PIL import Image
from one_hit_prediction import ModelLoader
from torchvision import models, transforms
import torch


coralmodel = ModelLoader()

@st.cache()



def prediction(image):
    transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.5, 0.5, 0.5],
        std = [0.5, 0.5, 0.5]
        )])
    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    ret_img = coralmodel.prediction(batch_t)
    return ret_img
    
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

def main():
    readme_text = st.markdown(read_markdown_file("UIEC2Net_readme.md"), unsafe_allow_html=True)
    image_upload = st.file_uploader("Upload an underwater image", type="jpg")
    if image_upload:
        image = Image.open(image_upload)
        st.image(image, caption = 'Uploaded Image.', use_column_width = True)
        st.write("")
        st.write("Just a second ...")
        ret_image = prediction(image_upload)
        st.image(ret_image, use_column_width=True)
        