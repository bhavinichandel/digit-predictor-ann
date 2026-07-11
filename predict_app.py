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

# 3. MNIST SCANNING & ERROR CONTROL TECHNIQUE
if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype("uint8"))
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)

    # Find all continuous white shapes (islands) on the board
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # COUNT CHECK: If more than one independent shape is detected (like 00 or 777)
    if len(contours) > 1:
        st.warning("⚠️ Multiple inputs detected! Please draw only a **single digit** at a time.")
        mnist_img = np.zeros((28, 28), dtype=np.uint8) # Keep preview blank
        allow_prediction = False
    elif len(contours) == 1:
        allow_prediction = True
        # Process the single digit normally
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped = gray[y:y+h, x:x+w]
        
        # Scale to 20x20 proportionally
        scale_factor = 20.0 / max(w, h)
        new_w = max(1, int(w * scale_factor))
        new_h = max(1, int(h * scale_factor))
        
        resized_digit = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Place perfectly in center of 28x28 matrix
        mnist_img = np.zeros((28, 28), dtype=np.uint8)
        cx = (28 - new_w) // 2
        cy = (28 - new_h) // 2
        mnist_img[cy:cy+new_h, cx:cx+new_w] = resized_digit
    else:
        allow_prediction = False
        mnist_img = np.zeros((28, 28), dtype=np.uint8)

    # Thicken lines slightly for optimization
    if len(contours) == 1:
        kernel = np.ones((2, 2), np.uint8)
        mnist_img = cv2.dilate(mnist_img, kernel, iterations=1)

    st.image(mnist_img, caption="Official 20x20 Proportional MNIST Preview", width=150)

    # 🧠 THE PREDICT BUTTON
    if st.button("🧠 Predict Digit"):
        if allow_prediction:
            normalized_img = mnist_img / 255.0
            final_input = np.expand_dims(normalized_img, axis=0)

            predictions = model.predict(final_input)
            best_guess = np.argmax(predictions)

            st.success(f"## 🎉 The drawn digit is: {best_guess}")
        else:
            st.error("❌ Cannot predict. Make sure you draw exactly one single digit clearly.")

    # 🧼 THE CLEAR CANVAS BUTTON
    if st.button("🧼 Clear Board"):
        st.session_state.canvas_key += 1
        st.rerun()
