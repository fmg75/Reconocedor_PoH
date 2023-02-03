import streamlit as st
from PIL import Image   
from utils import *
#import base64

# Logo PoH
#with open("logos/democratic-poh-logo-white.svg", "rb") as image_file:
#    logo_b64 = base64.b64encode(image_file.read()).decode()
#st.markdown(f'<img src="data:image/svg+xml;base64,{logo_b64}" style="float:right;">', unsafe_allow_html=True)



st.title('Esta Registrado en PoH ?')

def process_image(path):
    try:
        _models = FaceNetModels()
        img = Image.open(path)
        image_embedding = _models.embedding(_models.mtcnn(img))
        distancia = _models.Distancia(image_embedding)
        return distancia[0][0],distancia[1][0]
    except:
        return None

def upload_image():
    uploaded_file = st.file_uploader("Subir la imagen de un Humano y verificar si esta registrado en https://www.proofofhumanity.org/", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen subida', width=200)
        result = process_image(uploaded_file)
        if result:
            label, distance = result
            url = f"https://www.proofofhumanity.org/profile/{label}"
            st.markdown(f'<a href="{url}" target="_blank">https://www.proofofhumanity.org/profile/{label}</a>', unsafe_allow_html=True)
            st.write("Distancia Euclidiana: ", round(distance,4))
        else:
            st.write("Algo falló con la imagen, intenta con otra !!")

upload_image()
