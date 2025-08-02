import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("meme_identifier_model.h5")

# Preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Streamlit UI
st.title("🧠 Meme Identifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        prediction = model.predict(preprocess_image(image))[0][0]
        result = "✅ It's a Meme!" if prediction > 0.5 else "❌ Not a Meme"
        confidence = round(float(prediction) * 100, 2)

    st.markdown(f"### Result: {result}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
