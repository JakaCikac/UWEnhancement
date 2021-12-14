import streamlit as st
from pathlib import Path
from PIL import Image
from one_hit_prediction import ModelLoader
from torchvision import models, transforms
import torch


coralmodel = ModelLoader()

# @st.cache()


# @st.cache(hash_funcs={torch.nn.parameter.Parameter: prediction})
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
    batch_t = batch_t.cuda()
    ret_img = coralmodel.prediction(batch_t)
    return ret_img
    
def read_markdown_file(markdown_file):
    # return Path(markdown_file).read_text()
    with open(markdown_file, 'r', encoding='utf-8') as f:
        readme_lines = f.readlines()
        readme_buffer = []
        images = [
            'img/fig1.png',
            'img/image(2).png',
            'img/image(1).png',
            'img/image.png',
            'img/Screenshot2021-12-13at10.44.46.png',
            'img/22005133002_10708_68975_Sand.png',
            'img/22005133302_7173_47655_Macroalgae1.png',
            'img/22005134002_11809_75305_Siderastrea_siderea.png'
        ]
        for line in readme_lines:
            readme_buffer.append(line)
            for image in images:
                if image in line:
                    st.markdown(' '.join(readme_buffer[:-1]))
                    st.image(f'https://raw.githubusercontent.com/JakaCikac/UWEnhancement/streamlit-opt/{image}')
                    readme_buffer.clear()
        st.markdown(' '.join(readme_buffer))

def main():
    # readme_text = st.markdown(read_markdown_file("UIEC2Net_readme.md"), unsafe_allow_html=True)
    read_markdown_file("UIEC2Net_readme.md")
    image_upload = st.file_uploader("Upload an underwater image", type="jpg")
    if image_upload:
        image = Image.open(image_upload)
        st.image(image, caption = 'Uploaded Image.', use_column_width = True)
        st.write("")
        st.write("Just a second ...")
        ret_image = prediction(image_upload)
        st.image(ret_image, use_column_width=True)

if __name__ == "__main__":
    main()