import streamlit as st
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np

# Cargar el modelo
model = keras.models.load_model('fashion_mnist.keras')

# Etiquetas
classes = ["Camiseta/Top", "Pantalón", "Suéter", "Vestido", "Abrigo",
           "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botas"]

# Título de la aplicación
st.title("Clasificación de Prendas - Fashion MNIST")

# Cargar la imagen
uploaded_file = st.file_uploader("Sube una imagen de la prenda", type=["jpg", "png"])

if uploaded_file is not None:
    # Cargar y procesar la imagen
    image = Image.open(uploaded_file).convert("L")  # Escala de grises
    image = ImageOps.invert(image)  # Invertir colores para que las prendas sean blancas sobre fondo negro
    image = image.resize((28, 28))  # Redimensionar
    image = ImageOps.autocontrast(image)  # Ajustar contraste automáticamente
    st.image(image, caption="Imagen preprocesada (28x28)", use_container_width=True)

    # Convertir a array y normalizar
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predicción
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    # Mostrar resultados
    # st.write(f"Vector de predicción: {prediction}")
    st.subheader(f"Predicción: **{predicted_class}**")
