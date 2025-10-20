# data_augmentation_advanced_app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="🧩 Data Augmentation App", layout="wide")

st.title("🧠 Advanced Image Data Augmentation Web App")
st.write("Generate multiple augmented images with different transformations for deep learning datasets!")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img = np.array(img)
    st.image(img, caption="Original Image", use_container_width=True)

    st.sidebar.header("⚙️ Augmentation Controls")

    # User options
    num_images = st.sidebar.slider("🖼️ Number of Images to Generate", 1, 20, 5)
    fill_mode = st.sidebar.selectbox("🎨 Fill Mode (for rotation/zoom)", ["constant", "reflect", "nearest", "wrap"])
    rotation_range = st.sidebar.slider("↩️ Max Rotation (°)", 0, 45, 20)
    zoom_range = st.sidebar.slider("🔍 Zoom Range (%)", 50, 150, 100)
    brightness_range = st.sidebar.slider("💡 Brightness Range", 0.5, 2.0, 1.0)
    noise = st.sidebar.checkbox("🌫️ Add Gaussian Noise")
    blur = st.sidebar.checkbox("💧 Add Random Blur")

    st.subheader(f"✨ Generating {num_images} Augmented Images")

    augmented_images = []

    for i in range(num_images):
        aug_img = img.copy()

        # Random rotation
        angle = np.random.uniform(-rotation_range, rotation_range)
        (h, w) = aug_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply fill mode
        border_modes = {
            "constant": cv2.BORDER_CONSTANT,
            "reflect": cv2.BORDER_REFLECT,
            "nearest": cv2.BORDER_REPLICATE,
            "wrap": cv2.BORDER_WRAP
        }
        border_mode = border_modes[fill_mode]

        aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=border_mode)

        # Random zoom
        zoom_factor = np.random.uniform(zoom_range / 100, 1.0)
        zoomed = cv2.resize(aug_img, None, fx=zoom_factor, fy=zoom_factor)
        zh, zw = zoomed.shape[:2]
        if zoom_factor < 1.0:
            pad_x = (w - zw) // 2
            pad_y = (h - zh) // 2
            aug_img = cv2.copyMakeBorder(zoomed, pad_y, pad_y, pad_x, pad_x, border_mode)
        else:
            startx = (zw - w) // 2
            starty = (zh - h) // 2
            aug_img = zoomed[starty:starty + h, startx:startx + w]

        # Random brightness
        brightness = np.random.uniform(0.7, brightness_range)
        aug_img = cv2.convertScaleAbs(aug_img, alpha=brightness, beta=0)

        # Blur or noise
        if blur and np.random.rand() > 0.5:
            k = np.random.choice([3, 5])
            aug_img = cv2.GaussianBlur(aug_img, (k, k), 0)

        if noise:
            noise_img = np.random.normal(0, 15, aug_img.shape).astype(np.uint8)
            aug_img = cv2.add(aug_img, noise_img)

        augmented_images.append(aug_img)

    # Display augmented images
    cols = st.columns(5)
    for i, aug_img in enumerate(augmented_images):
        with cols[i % 5]:
            st.image(aug_img, caption=f"Image {i+1}", use_container_width=True)

    # Zip and download all images
    output = io.BytesIO()
    from zipfile import ZipFile
    with ZipFile(output, "w") as zipf:
        for i, aug_img in enumerate(augmented_images):
            img_pil = Image.fromarray(aug_img)
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format="PNG")
            zipf.writestr(f"augmented_{i+1}.png", img_bytes.getvalue())

    st.download_button(
        "💾 Download All Augmented Images (ZIP)",
        data=output.getvalue(),
        file_name="augmented_images.zip",
        mime="application/zip"
    )

else:
    st.info("👆 Upload an image to start augmenting!")

st.markdown("---")
st.markdown("**Created by [Boinapalli Ramya] | Full Stack Data Scientist | OpenCV + Streamlit + AI Vision**")
