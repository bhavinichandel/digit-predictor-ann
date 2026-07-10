import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 1. LOAD THE ANN MODEL BRAIN
model = tf.keras.models.load_model("digit_model.h5")

st.title("🔢 ANN Handwritten Digit Predictor")
st.write("Draw a single digit (0-9) inside the box below!")

# 2. CANVAS DRAWING BOARD
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",
)

# 3. OFFICIAL MNIST SCANNING TECHNIQUE
if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype("uint8"))
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)

    # Find the pixel coordinates where you actually drew
    pts = np.argwhere(gray > 0)
    
    if len(pts) > 0:
        # Crop tight around the digit
        y_min, x_min = pts.min(axis=0)
        y_max, x_max = pts.max(axis=0)
        cropped = gray[y_min:y_max+1, x_min:x_max+1]
        
        # A. Scale the digit proportionally to fit inside a 20x20 box (MNIST Standard!)
        h, w = cropped.shape
        scale_factor = 20.0 / max(h, w)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        # Make sure dimensions are at least 1 pixel
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        resized_digit = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # B. Create a blank 28x28 black canvas
        mnist_img = np.zeros((28, 28), dtype=np.uint8)
        
        # C. Place the 20x20 digit exactly in the true middle of the 28x28 image
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        mnist_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_digit
    else:
        # If nothing is drawn, show a blank 28x28 frame
        mnist_img = np.zeros((28, 28), dtype=np.uint8)

    # Thicken lines slightly so the neural switches can read it smoothly
    kernel = np.ones((2, 2), np.uint8)
    mnist_img = cv2.dilate(mnist_img, kernel, iterations=1)

    st.image(mnist_img, caption="Official 20x20 Proportional MNIST Preview", width=150)

    # 🧠 THE PREDICT BUTTON
    if st.button("🧠 Predict Digit"):
        normalized_img = mnist_img / 255.0
        final_input = np.expand_dims(normalized_img, axis=0)

        predictions = model.predict(final_input)
        best_guess = np.argmax(predictions)

        st.success(f"## 🎉 The ANN Model predicts: {best_guess}")

    # 🧼 THE CLEAR CANVAS BUTTON
    if st.button("🧼 Clear Board"):
        st.session_state.canvas_key += 1
        st.rerun()