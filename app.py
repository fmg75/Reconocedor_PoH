import streamlit as st
from PIL import Image   
from utils import *

st.title('Registrado en Proof of Humanity?')

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
    uploaded_file = st.file_uploader("Subir la imagen de un Humano y verificar si esta registrado en https://www.proofofhumanity.org/ o https://app.proofofhumanity.id/", type=["jpg", "png"])
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
            st.write("Algo falló con la imagen proporcionada, intenta con otra !!")

 # información adicional
    with st.expander('Información adicional'):
         st.write('Esta aplicación utiliza el modelo de redes neuronales conocido como ResNet '+
                 'para reconocer características de rostros en imágenes. Con esta tecnología se construyó un diccionario que la app utiliza ' + 
                 'para comparar con las caracteristicas del rostro ingresadas por el usuario. '+     
                 'El usuario puede subir una imagen desde su dispositivo o utilizar la cámara para tomar una foto, y la aplicación '+ 
                 'devolverá el perfil de PoH más similar al de la base de datos junto con la distancia euclidiana entre los dos rostros.' +
                 ' Si la imagen corresponde a un humano registrado el rostro sera reconocido en correspondencia con una distacia euclidiana muy baja, proxima a cero.'
                 ' Por el momento se reconocen 16K de registrados en PoH, la base de datos se ira actualizando cada mil registrados nuevos.') 

upload_image()
