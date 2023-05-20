import streamlit as st
import io
from PIL import Image
from utils import *

# Configura el favicon de la aplicación
favicon = open("logos/favicon.ico", "rb").read()
st.set_page_config(page_title="Registrado en PoH?", page_icon=favicon)

# agrego logo
logo = Image.open("logos/democratic-poh-logo-text-hi-res-p-500.png")
st.image(logo, use_column_width=True)

# Titulo en color verde
st.markdown(
    "<h1 style='color: green;'>Registered with Proof of Humanity?</h1>",
    unsafe_allow_html=True,
)


def process_image(path):
    try:
        _models = FaceNetModels()
        img = Image.open(path)

        # Verificar si la imagen está en formato PNG y convertir a JPG si es necesario
        if img.format == "PNG":
            jpg_io = (
                io.BytesIO()
            )  # Crear un objeto BytesIO para guardar la imagen en memoria
            img = img.convert(
                "RGB"
            )  # Convertir a modo RGB (requerido para guardar como JPG)
            img.save(
                jpg_io, format="JPEG"
            )  # Guardar la imagen en el objeto BytesIO en formato JPG
            jpg_io.seek(0)  # Colocar el puntero de lectura al inicio del objeto BytesIO
            img = Image.open(
                jpg_io
            )  # Abrir la imagen en formato JPG desde el objeto BytesIO

        image_embedding = _models.embedding(_models.mtcnn(img))
        return _models.Distancia(image_embedding)
    except:
        return None


def upload_image():
    uploaded_file = st.file_uploader(
        "Upload the image of a Human and verify if it is registered in https://app.proofofhumanity.id/",
        type=["jpg", "png"],
    )
    print(uploaded_file)
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="uploaded image ", width=200)
        result = process_image(uploaded_file)
        if result:
            label, distance = result
            url = f"https://app.proofofhumanity.id/profile/{label}"
            st.markdown(
                f'<a href="{url}" target="_blank">https://app.proofofhumanity.id/profile/{label}</a>',
                unsafe_allow_html=True,
            )
            st.write("Euclidean Distance: ", round(distance, 4))
        else:
            st.write(
                "Something went wrong with the provided image, please try another!!"
            )

    # información adicional
    with st.expander("Información adicional"):
        st.write(
            "This application uses the neural network model known as ResNet"
            + "to recognize features of faces in images. With this technology, a dictionary was built that the app uses "
            + "to compare with the features of the face entered by the user. "
            + "The user can upload an image from their device or use the camera to take a photo, and the application "
            + "will return the closest PoH profile to the one in the database along with the Euclidean distance between the two faces."
            + "If the image corresponds to a registered human, the face will be recognized in correspondence with a very low Euclidean distance, close to zero."
            "At the moment 17 thousand registered in PoH are recognized, the database will be updated every thousand new registered."
        )


# lanza app
upload_image()
