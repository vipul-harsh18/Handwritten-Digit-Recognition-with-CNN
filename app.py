
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="",
    layout="centered"
)


st.markdown("""
<style>
.stApp {
    background-color: #111827;
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

.stButton > button {
    background-color: #6C63FF;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #5148e5;
    color: white;
}

[data-testid="stFileUploader"] {
    background-color: #1F2937;
    border-radius: 10px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return keras.models.load_model("digit_model.keras")


model = load_model()


def prepare_image(image, invert=True):
    image = ImageOps.exif_transpose(image).convert("L")
    arr = np.array(image)

    if invert:
        border = np.concatenate((
            arr[0, :],
            arr[-1, :],
            arr[:, 0],
            arr[:, -1]
        ))

        if np.median(border) > 127:
            arr = 255 - arr

    mask = arr > 35

    if not np.any(mask):
        return None, None

    ys, xs = np.where(mask)

    digit = arr[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    h, w = digit.shape

    scale = 20 / max(h, w)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    digit_image = Image.fromarray(digit).resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new("L", (28, 28), 0)

    x = (28 - new_w) // 2
    y = (28 - new_h) // 2

    canvas.paste(digit_image, (x, y))

    input_array = np.asarray(
        canvas,
        dtype=np.float32
    ) / 255.0

    input_array = input_array.reshape(1, 28, 28, 1)

    return canvas, input_array


def show_prediction(input_data, processed_image):
    predictions = model.predict(input_data, verbose=0)[0]

    predicted_digit = int(np.argmax(predictions))
    confidence = float(predictions[predicted_digit]) * 100

    st.success(f"Predicted Digit: {predicted_digit}")

    st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%"
    )

    st.subheader("Processed Image")

    st.image(
        processed_image,
        width=180,
        caption="28 × 28 image given to the CNN"
    )

    st.subheader("Prediction Probabilities")

    chart_data = {
        "Digit": [str(i) for i in range(10)],
        "Probability": predictions
    }

    st.bar_chart(
        chart_data,
        x="Digit",
        y="Probability"
    )


st.title("Handwritten Digit Recognition")

st.write(
    "Upload an image or draw a digit from 0 to 9. "
    "The CNN model will predict the digit."
)


tab1, tab2 = st.tabs([
    "Upload Image",
    "Draw Digit"
])


with tab1:
    st.subheader("Upload a Handwritten Digit")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            width=250
        )

        if st.button("Predict Digit", key="upload_predict"):
            processed_image, input_data = prepare_image(
                image,
                invert=True
            )

            if input_data is None:
                st.warning(
                    "No digit detected. Please upload a clear image."
                )
            else:
                show_prediction(input_data, processed_image)


with tab2:
    st.subheader("Draw a Digit")

    st.write(
        "Draw one digit using your mouse or touch screen."
    )

    
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="draw_digit",
        update_streamlit=True,
        return_image_data=True
    )


    if st.button("Predict Digit", key="draw_predict"):
        try:
            drawing = canvas_result.image_data

            if drawing is None:
                st.warning(
                    "Draw a digit on the canvas first."
                )
            else:
                rgb = drawing[:, :, :3].astype(np.uint8)

                gray_image = Image.fromarray(rgb).convert("L")

                processed_image, input_data = prepare_image(
                    gray_image,
                    invert=False
                )

                if input_data is None:
                    st.warning(
                        "Please draw a digit first."
                    )
                else:
                    show_prediction(
                        input_data,
                        processed_image
                    )

        except Exception as e:
            st.error(f"Canvas error: {e}")
