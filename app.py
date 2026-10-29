import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            "digit_model.keras",
            compile=False
        )
        return model

    except Exception as e:
        st.error("❌ Model could not be loaded.")
        st.error(f"Error: {e}")
        st.stop()


model = load_model()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🔢 Handwritten Digit Recognition")

st.markdown(
    """
    ### CNN-Based Deep Learning Project

    Draw a handwritten digit from **0 to 9** in the canvas below.
    The trained Convolutional Neural Network will predict the digit.
    """
)

st.divider()


# --------------------------------------------------
# TWO COLUMNS
# --------------------------------------------------

col1, col2 = st.columns(2)


# --------------------------------------------------
# DRAWING CANVAS
# --------------------------------------------------

with col1:

    st.subheader("✍️ Draw Your Digit")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

    predict_button = st.button(
        "🔮 Predict Digit",
        use_container_width=True
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

with col2:

    st.subheader("🤖 Prediction")

    if predict_button:

        if canvas_result.image_data is not None:

            # Get canvas image
            image = canvas_result.image_data

            # Convert RGBA to grayscale
            image = Image.fromarray(
                image.astype("uint8")
            ).convert("L")

            # Resize to MNIST size
            image = image.resize((28, 28))

            # Convert image to NumPy array
            image_array = np.array(image)

            # Normalize pixel values
            image_array = image_array.astype(
                "float32"
            ) / 255.0

            # Reshape for CNN
            image_array = image_array.reshape(
                1, 28, 28, 1
            )

            # Make prediction
            prediction = model.predict(
                image_array,
                verbose=0
            )[0]

            # Get predicted digit
            predicted_digit = int(
                np.argmax(prediction)
            )

            # Get confidence
            confidence = float(
                prediction[predicted_digit] * 100
            )

            # Display prediction
            st.success(
                f"Predicted Digit: {predicted_digit}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            # Prediction probabilities
            st.subheader(
                "📊 Prediction Probabilities"
            )

            probabilities = {
                str(i): float(prediction[i])
                for i in range(10)
            }

            st.bar_chart(probabilities)

        else:

            st.warning(
                "⚠️ Please draw a digit first."
            )


# --------------------------------------------------
# ABOUT MODEL
# --------------------------------------------------

st.divider()

st.subheader("🧠 About the Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset", "MNIST")

with col2:
    st.metric("Image Size", "28 × 28")

with col3:
    st.metric("Classes", "10")


st.info(
    """
    This application uses a Convolutional Neural Network (CNN)
    trained on the MNIST handwritten digit dataset.
    The model classifies handwritten digits from 0 to 9.
    """
)
