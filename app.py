import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def predict_class(image):
    RGBImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    RGBImg = cv2.resize(RGBImg, (224, 224))
    plt.imshow(RGBImg)
    plt.axis('off')
    plt.show()
    image = np.array(RGBImg) / 255.0
    new_model = tf.keras.models.load_model("64x3-CNN.model", compile=False)
    predict = new_model.predict(np.array([image]))
    per = np.argmax(predict, axis=1)
    if per == 1:
        prediction = 'Diabetic Retinopathy Not Detected'
    else:
        prediction = 'Diabetic Retinopathy Detected'

    return prediction


def main():
    st.title("Diabetic Retinopathy Prediction")
    st.text("Upload an image for prediction")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Predict'):
            prediction = predict_class(image)
            st.markdown(f"<h1 style='text-align: center; color: blue;'>{prediction}</h1>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
