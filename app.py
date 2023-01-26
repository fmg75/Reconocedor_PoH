import streamlit as st
from PIL import Image   
from utils import *

st.title('Esta Registrado en PoH ?')

def process_image(path):
    face_net_models = FaceNetModels()
    img = Image.open(path)
    image_embedding = face_net_models.embedding(face_net_models.mtcnn(img))
    distancia = face_net_models.Distancia(image_embedding)
    return distancia[0][0],distancia[1][0]

def upload_image():
    uploaded_file = st.file_uploader("Subir la imagen de un Humano y se verificara si esta registrado en https://www.proofofhumanity.org/", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        #st.image(image, caption='Imagen subida', use_column_width=True)
        st.image(image, caption='Imagen subida', width=200)
        label, distance = process_image(uploaded_file)

        url = f"https://www.proofofhumanity.org/profile/{label}"
        st.markdown(f'<a href="{url}" target="_blank">https://www.proofofhumanity.org/profile/{label}</a>', unsafe_allow_html=True)
        st.write("Distancia Euclidiana: ", distance)

upload_image()
