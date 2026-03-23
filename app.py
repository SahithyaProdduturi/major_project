# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from PIL import Image
# import matplotlib.pyplot as plt

# # -------------------------------
# # Load Model
# # -------------------------------
# model = load_model("covid_xray_coroNet_model.h5")

# class_names = ['COVID', 'Normal', 'Pneumonia']

# IMG_SIZE = (224, 224)

# # -------------------------------
# # Preprocess Image
# # -------------------------------
# def preprocess_image(img):
#     img = img.resize(IMG_SIZE)
#     img_array = np.array(img)
#     img_array = img_array / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # -------------------------------
# # Prediction
# # -------------------------------
# def predict_image(img):
#     img_array = preprocess_image(img)
#     predictions = model.predict(img_array)
    
#     predicted_index = np.argmax(predictions[0])
#     predicted_class = class_names[predicted_index]
#     confidence = predictions[0][predicted_index] * 100
    
#     return predicted_class, confidence, img_array

# # -------------------------------
# # Grad-CAM
# # -------------------------------
# def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

#     grad_model = tf.keras.models.Model(
#         inputs=model.input,
#         outputs=[
#             model.get_layer(last_conv_layer_name).output,
#             model.output
#         ]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)

#         # If predictions is list → take first element
#         if isinstance(predictions, list):
#             predictions = predictions[0]

#         # Convert to numpy safely
#         preds = predictions[0].numpy()

#         pred_index = np.argmax(preds)   # pure integer

#         class_channel = predictions[:, pred_index]

#     grads = tape.gradient(class_channel, conv_outputs)

#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     conv_outputs = conv_outputs[0]

#     heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

#     heatmap = tf.maximum(heatmap, 0)

#     # Avoid division by zero
#     if tf.reduce_max(heatmap) != 0:
#         heatmap /= tf.reduce_max(heatmap)

#     return heatmap.numpy()
# # -------------------------------
# # Streamlit UI
# # -------------------------------
# st.title("🩺 AI-Based Chest X-ray Disease Detection")
# st.write("Upload a chest X-ray image to detect COVID / Normal / Pneumonia")

# uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
    
#     img = Image.open(uploaded_file).convert("RGB")
    
#     st.image(img, caption="Uploaded X-ray", use_column_width=True)

#     if st.button("Analyze Image"):
        
#         predicted_class, confidence, img_array = predict_image(img)

#         st.subheader("Prediction Result")
#         st.write(f"**Disease:** {predicted_class}")
#         st.write(f"**Confidence:** {confidence:.2f}%")

#         # Grad-CAM
#         heatmap = make_gradcam_heatmap(
#             img_array,
#             model,
#             last_conv_layer_name="block14_sepconv2_act"
#         )

#         # Overlay
#         img_np = np.array(img.resize(IMG_SIZE))
#         heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
#         heatmap_resized = np.uint8(255 * heatmap_resized)
#         heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

#         superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

#         st.subheader("Highlighted Affected Regions")
#         st.image(superimposed_img, use_column_width=True)

# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# from PIL import Image

# # -------------------------------
# # Load Model (cache for speed)
# # -------------------------------
# @st.cache_resource
# def load_trained_model():
#     return load_model("covid_xray_coroNet_model.h5")

# model = load_trained_model()

# class_names = ['COVID', 'Normal', 'Pneumonia']
# IMG_SIZE = (224, 224)

# # -------------------------------
# # Preprocess Image
# # -------------------------------
# def preprocess_image(img):
#     img = img.resize(IMG_SIZE)
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # -------------------------------
# # Prediction
# # -------------------------------
# def predict_image(img):
#     img_array = preprocess_image(img)
#     predictions = model.predict(img_array)
    
#     predicted_index = np.argmax(predictions[0])
#     predicted_class = class_names[predicted_index]
#     confidence = predictions[0][predicted_index] * 100
    
#     return predicted_class, confidence, img_array

# # -------------------------------
# # Grad-CAM
# # -------------------------------
# def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

#     grad_model = tf.keras.models.Model(
#         inputs=model.input,
#         outputs=[
#             model.get_layer(last_conv_layer_name).output,
#             model.output
#         ]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)

#         if isinstance(predictions, list):
#             predictions = predictions[0]

#         preds = predictions[0].numpy()
#         pred_index = np.argmax(preds)

#         class_channel = predictions[:, pred_index]

#     grads = tape.gradient(class_channel, conv_outputs)

#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]

#     heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
#     heatmap = tf.maximum(heatmap, 0)

#     if tf.reduce_max(heatmap) != 0:
#         heatmap /= tf.reduce_max(heatmap)

#     return heatmap.numpy()

# # -------------------------------
# # Streamlit UI
# # -------------------------------
# st.title("🩺 AI-Based Chest X-ray Disease Detection")
# st.write("Upload a chest X-ray image to detect COVID / Normal / Pneumonia")

# uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:

#     img = Image.open(uploaded_file).convert("RGB")
#     st.image(img, caption="Uploaded X-ray", use_container_width=True)

#     if st.button("Analyze Image"):

#         predicted_class, confidence, img_array = predict_image(img)

#         st.subheader("Prediction Result")
#         st.write(f"**Disease:** {predicted_class}")
#         st.write(f"**Confidence:** {confidence:.2f}%")

#         # Generate Grad-CAM heatmap
#         heatmap = make_gradcam_heatmap(
#             img_array,
#             model,
#             last_conv_layer_name="block14_sepconv2_act"
#         )

#         # Resize original image
#         img_np = np.array(img.resize(IMG_SIZE))

#         # Resize heatmap
#         heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
#         heatmap_resized = np.uint8(255 * heatmap_resized)

#         # -----------------------------
#         # Show ONLY high-activation areas
#         # -----------------------------
#         threshold = 180  # Adjust (150–200)

#         mask = np.zeros_like(heatmap_resized)
#         mask[heatmap_resized > threshold] = 255

#         # Create red mask
#         red_mask = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
#         red_mask[:, :, 2] = mask  # Red channel

#         # Overlay red mask
#         superimposed_img = cv2.addWeighted(img_np, 1.0, red_mask, 0.6, 0)

#         st.subheader("Highlighted High-Activation Regions")
#         st.image(superimposed_img, use_container_width=True)


# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# from PIL import Image

# # -------------------------------
# # Page Config
# # -------------------------------
# st.set_page_config(page_title="AI X-ray Detection", layout="wide")

# # -------------------------------
# # Load Model
# # -------------------------------
# @st.cache_resource
# def load_trained_model():
#     return load_model("covid_xray_coroNet_model.h5")

# model = load_trained_model()

# class_names = ['COVID', 'Normal', 'Pneumonia']
# IMG_SIZE = (224, 224)

# # -------------------------------
# # Preprocess Image
# # -------------------------------
# def preprocess_image(img):
#     img = img.resize(IMG_SIZE)
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # -------------------------------
# # Prediction
# # -------------------------------
# def predict_image(img):
#     img_array = preprocess_image(img)
#     predictions = model.predict(img_array)

#     predicted_index = np.argmax(predictions[0])
#     predicted_class = class_names[predicted_index]
#     confidence = predictions[0][predicted_index] * 100

#     return predicted_class, confidence, img_array, predictions[0]

# # -------------------------------
# # Grad-CAM
# # -------------------------------
# def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

#     grad_model = tf.keras.models.Model(
#         inputs=model.input,
#         outputs=[
#             model.get_layer(last_conv_layer_name).output,
#             model.output
#         ]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)

#         if isinstance(predictions, list):
#             predictions = predictions[0]

#         pred_index = tf.argmax(predictions[0])
#         class_channel = predictions[:, pred_index]

#     grads = tape.gradient(class_channel, conv_outputs)

#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]

#     heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
#     heatmap = tf.maximum(heatmap, 0)

#     if tf.reduce_max(heatmap) != 0:
#         heatmap /= tf.reduce_max(heatmap)

#     return heatmap.numpy()

# # -------------------------------
# # UI HEADER
# # -------------------------------
# st.markdown("<h1 style='text-align: center;'>🩺 AI Chest X-ray Analyzer</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center;'>Detect COVID-19, Pneumonia, or Normal using Deep Learning + Explainable AI</p>", unsafe_allow_html=True)

# # -------------------------------
# # File Upload
# # -------------------------------
# uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

# # -------------------------------
# # Threshold Slider
# # -------------------------------
# threshold = st.slider("🔥 Highlight Sensitivity (Grad-CAM)", 100, 255, 180)

# # -------------------------------
# # Main Logic
# # -------------------------------
# if uploaded_file is not None:

#     img = Image.open(uploaded_file).convert("RGB")

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("📷 Original Image")
#         st.image(img, use_container_width=True)

#     if st.button("🚀 Analyze Image"):

#         with st.spinner("Analyzing..."):

#             predicted_class, confidence, img_array, raw_preds = predict_image(img)

#             # ---------------------------
#             # Prediction Output
#             # ---------------------------
#             st.subheader("🧠 Prediction Result")

#             st.success(f"Prediction: {predicted_class}")
#             st.progress(int(confidence))

#             st.write(f"Confidence: **{confidence:.2f}%**")

#             # Show all class probabilities
#             st.write("### 📊 Class Probabilities")
#             for i, cls in enumerate(class_names):
#                 st.write(f"{cls}: {raw_preds[i]*100:.2f}%")

#             # ---------------------------
#             # Grad-CAM
#             # ---------------------------
#             heatmap = make_gradcam_heatmap(
#                 img_array,
#                 model,
#                 last_conv_layer_name="block14_sepconv2_act"
#             )

#             img_np = np.array(img.resize(IMG_SIZE))

#             heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
#             heatmap_resized = np.uint8(255 * heatmap_resized)

#             # High activation mask
#             mask = np.zeros_like(heatmap_resized)
#             mask[heatmap_resized > threshold] = 255

#             # Red overlay
#             red_mask = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
#             red_mask[:, :, 2] = mask

#             superimposed_img = cv2.addWeighted(img_np, 1.0, red_mask, 0.6, 0)

#             with col2:
#                 st.subheader("🔥 Affected Regions")
#                 st.image(superimposed_img, use_container_width=True)

#             # ---------------------------
#             # Explanation Box
#             # ---------------------------
#             st.info("""
# 🔍 **How to interpret:**
# - Red regions = model focuses here
# - Brighter red = stronger influence
# - Only high-risk areas are shown (filtered)
# """)

# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# from PIL import Image

# # -------------------------------
# # Page Config
# # -------------------------------
# st.set_page_config(page_title="AI X-ray Detection", layout="wide")

# # -------------------------------
# # Load Model
# # -------------------------------
# @st.cache_resource
# def load_trained_model():
#     return load_model("covid_xray_coroNet_model.h5")

# model = load_trained_model()

# class_names = ['COVID', 'Normal', 'Pneumonia']
# IMG_SIZE = (224, 224)

# # -------------------------------
# # Preprocess Image
# # -------------------------------
# def preprocess_image(img):
#     img = img.resize(IMG_SIZE)
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # -------------------------------
# # Prediction
# # -------------------------------
# def predict_image(img):
#     img_array = preprocess_image(img)
#     predictions = model.predict(img_array)

#     predicted_index = np.argmax(predictions[0])
#     predicted_class = class_names[predicted_index]
#     confidence = predictions[0][predicted_index] * 100

#     return predicted_class, confidence, img_array, predictions[0]

# # -------------------------------
# # Grad-CAM
# # -------------------------------
# def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

#     grad_model = tf.keras.models.Model(
#         inputs=model.input,
#         outputs=[
#             model.get_layer(last_conv_layer_name).output,
#             model.output
#         ]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)

#         if isinstance(predictions, list):
#             predictions = predictions[0]

#         pred_index = tf.argmax(predictions[0])
#         class_channel = predictions[:, pred_index]

#     grads = tape.gradient(class_channel, conv_outputs)

#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]

#     heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
#     heatmap = tf.maximum(heatmap, 0)

#     if tf.reduce_max(heatmap) != 0:
#         heatmap /= tf.reduce_max(heatmap)

#     return heatmap.numpy()

# # -------------------------------
# # UI HEADER
# # -------------------------------
# st.markdown("<h1 style='text-align: center;'>🩺 AI Chest X-ray Analyzer</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center;'>COVID-19 | Pneumonia | Normal Detection with Explainable AI</p>", unsafe_allow_html=True)

# # -------------------------------
# # File Upload
# # -------------------------------
# uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

# # -------------------------------
# # Main Logic
# # -------------------------------
# if uploaded_file is not None:

#     img = Image.open(uploaded_file).convert("RGB")

#     threshold = st.slider("🔥 Highlight Sensitivity (Grad-CAM)", 100, 255, 180)

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("📷 Original Image")
#         st.image(img, use_container_width=True)

#     if st.button("🚀 Analyze Image"):

#         with st.spinner("Analyzing..."):

#             predicted_class, confidence, img_array, raw_preds = predict_image(img)

#             # ---------------------------
#             # Prediction Output
#             # ---------------------------
#             st.subheader("🧠 Prediction Result")
#             st.success(f"Prediction: {predicted_class}")
#             st.progress(int(confidence))
#             st.write(f"Confidence: **{confidence:.2f}%**")

#             st.write("### 📊 Class Probabilities")
#             for i, cls in enumerate(class_names):
#                 st.write(f"{cls}: {raw_preds[i]*100:.2f}%")

#             # ---------------------------
#             # Grad-CAM
#             # ---------------------------
#             heatmap = make_gradcam_heatmap(
#                 img_array,
#                 model,
#                 last_conv_layer_name="block14_sepconv2_act"
#             )

#             img_np = np.array(img.resize(IMG_SIZE))

#             heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
#             heatmap_resized = np.uint8(255 * heatmap_resized)

#             # 🌈 Full Heatmap
#             heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
#             overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

#             # 🔥 High activation mask
#             mask = np.zeros_like(heatmap_resized)
#             mask[heatmap_resized > threshold] = 255

#             # Use RED correctly
#             red_mask = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
#             red_mask[:, :, 0] = 0   # Blue
#             red_mask[:, :, 1] = 0   # Green
#             red_mask[:, :, 2] = mask  # Red

#             filtered_overlay = cv2.addWeighted(img_np, 1.0, red_mask, 0.7, 0)

#             # ✅ FIX: Convert BGR → RGB
#             overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
#             filtered_overlay = cv2.cvtColor(filtered_overlay, cv2.COLOR_BGR2RGB)

#             # ---------------------------
#             # Display Outputs
#             # ---------------------------
#             col2_1, col2_2 = st.columns(2)

#             with col2_1:
#                 st.subheader("🌈 Full Heatmap")
#                 st.image(overlay, use_container_width=True)

#             with col2_2:
#                 st.subheader("🔥 High Activation Regions")
#                 st.image(filtered_overlay, use_container_width=True)

#             # ---------------------------
#             # Legend
#             # ---------------------------
#             st.markdown("### 🎨 Heatmap Color Legend")
#             st.markdown("""
# - 🔵 **Blue** → Low importance  
# - 🟢 **Green** → Medium importance  
# - 🔴 **Red** → High importance (critical regions)  

# 👉 Right image shows ONLY high-importance regions.
# """)

#             # ---------------------------
#             # Info
#             # ---------------------------
#             st.info("""
# 🔍 **Interpretation Guide:**
# - Model highlights areas used for prediction  
# - Red zones = strongest attention  
# - Helps doctors understand AI reasoning  
# """)

# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# from PIL import Image

# # -------------------------------
# # Page Config
# # -------------------------------
# st.set_page_config(page_title="AI X-ray Analyzer", layout="wide")

# # -------------------------------
# # Dark Theme Styling
# # -------------------------------
# st.markdown("""
# <style>
# body {
#     background-color: #0e1117;
#     color: white;
# }
# </style>
# """, unsafe_allow_html=True)

# # -------------------------------
# # Load Model
# # -------------------------------
# @st.cache_resource
# def load_trained_model():
#     return load_model("covid_xray_coroNet_model.h5")

# model = load_trained_model()

# class_names = ['COVID', 'Normal', 'Pneumonia']
# IMG_SIZE = (224, 224)

# # -------------------------------
# # Sidebar
# # -------------------------------
# with st.sidebar:
#     st.header("⚙️ Settings")
#     threshold = st.slider("Grad-CAM Sensitivity", 100, 255, 180)
#     st.markdown("---")
    

# # -------------------------------
# # Preprocess
# # -------------------------------
# def preprocess_image(img):
#     img = img.resize(IMG_SIZE)
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # -------------------------------
# # Prediction
# # -------------------------------
# def predict_image(img):
#     img_array = preprocess_image(img)
#     predictions = model.predict(img_array)

#     predicted_index = np.argmax(predictions[0])
#     predicted_class = class_names[predicted_index]
#     confidence = predictions[0][predicted_index] * 100

#     return predicted_class, confidence, img_array, predictions[0]

# # -------------------------------
# # Grad-CAM
# # -------------------------------
# def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

#     grad_model = tf.keras.models.Model(
#         inputs=model.input,
#         outputs=[
#             model.get_layer(last_conv_layer_name).output,
#             model.output
#         ]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)

#         if isinstance(predictions, list):
#             predictions = predictions[0]

#         pred_index = tf.argmax(predictions[0])
#         class_channel = predictions[:, pred_index]

#     grads = tape.gradient(class_channel, conv_outputs)

#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]

#     heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
#     heatmap = tf.maximum(heatmap, 0)

#     if tf.reduce_max(heatmap) != 0:
#         heatmap /= tf.reduce_max(heatmap)

#     return heatmap.numpy()

# # -------------------------------
# # Header
# # -------------------------------
# st.markdown("<h1 style='text-align: center;'>🩺 AI Chest X-ray Analyzer</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center;'>COVID-19 | Pneumonia | Normal Detection with Explainable AI</p>", unsafe_allow_html=True)

# st.warning("⚠️ This tool is for educational purposes only. Not a medical diagnosis.")

# # -------------------------------
# # Upload
# # -------------------------------
# uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:

#     img = Image.open(uploaded_file).convert("RGB")

#     st.image(img, caption="Uploaded Image", use_container_width=True)

#     if st.button("🚀 Analyze Image"):

#         with st.spinner("Analyzing..."):

#             predicted_class, confidence, img_array, raw_preds = predict_image(img)

#             # ---------------------------
#             # Prediction Card
#             # ---------------------------
#             st.markdown(f"""
#             <div style="
#                 background-color:#1e1e1e;
#                 padding:20px;
#                 border-radius:10px;
#                 text-align:center;
#                 color:white;">
#                 <h2>🧠 {predicted_class}</h2>
#                 <h4>Confidence: {confidence:.2f}%</h4>
#             </div>
#             """, unsafe_allow_html=True)

#             # ---------------------------
#             # Tabs
#             # ---------------------------
#             tab1, tab2, tab3 = st.tabs(["📊 Results", "🔥 Heatmaps", "ℹ️ Info"])

#             # ---------------------------
#             # TAB 1: Results
#             # ---------------------------
#             with tab1:
#                 st.subheader("📊 Class Probabilities")
#                 st.bar_chart({
#                     "COVID": raw_preds[0],
#                     "Normal": raw_preds[1],
#                     "Pneumonia": raw_preds[2]
#                 })

#             # ---------------------------
#             # Grad-CAM Processing
#             # ---------------------------
#             heatmap = make_gradcam_heatmap(
#                 img_array,
#                 model,
#                 last_conv_layer_name="block14_sepconv2_act"
#             )

#             img_np = np.array(img.resize(IMG_SIZE))

#             heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
#             heatmap_resized = np.uint8(255 * heatmap_resized)

#             # Full Heatmap
#             heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
#             overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

#             # High Activation Mask
#             mask = np.zeros_like(heatmap_resized)
#             mask[heatmap_resized > threshold] = 255

#             red_mask = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
#             red_mask[:, :, 2] = mask

#             filtered_overlay = cv2.addWeighted(img_np, 1.0, red_mask, 0.7, 0)

#             # Convert BGR → RGB
#             overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
#             filtered_overlay = cv2.cvtColor(filtered_overlay, cv2.COLOR_BGR2RGB)

#             # ---------------------------
#             # TAB 2: Heatmaps
#             # ---------------------------
#             with tab2:
#                 st.subheader("🧾 Visual Analysis")

#                 colA, colB, colC = st.columns(3)

#                 with colA:
#                     st.image(img, caption="Original")

#                 with colB:
#                     st.image(overlay, caption="Full Heatmap")

#                 with colC:
#                     st.image(filtered_overlay, caption="Affected Regions")

#                 st.markdown("### 🎨 Heatmap Legend")
#                 st.markdown("""
# - 🔵 Blue → Low importance  
# - 🟢 Green → Medium importance  
# - 🔴 Red → High importance  
# """)

#             # ---------------------------
#             # TAB 3: Info
#             # ---------------------------
#             with tab3:
#                 st.markdown("### 🤖 Model Details")
#                 st.write("""
# - Model: CoroNet (Xception-based)
# - Input Size: 224x224
# - Technique: Grad-CAM Explainability
# - Classes: COVID, Pneumonia, Normal
# """)

#                 st.markdown("### 🔍 Interpretation")
#                 st.write("""
# - Red regions = strong model attention  
# - Helps identify infected lung areas  
# - Improves trust in AI predictions  
# """)

# # -------------------------------
# # Footer
# # -------------------------------
# st.markdown("---")
# st.markdown("👨‍⚕️ Developed as AI-based Medical Diagnosis System | Final Year Project")

# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# from PIL import Image

# # -------------------------------
# # Page Config
# # -------------------------------
# st.set_page_config(page_title="AI X-ray Analyzer", layout="wide")

# # -------------------------------
# # Styling
# # -------------------------------
# st.markdown("""
# <style>
# body {
#     background-color: #0e1117;
#     color: white;
# }
# </style>
# """, unsafe_allow_html=True)

# # -------------------------------
# # Load Model
# # -------------------------------
# @st.cache_resource
# def load_trained_model():
#     return load_model("covid_xray_coroNet_model.h5")

# model = load_trained_model()

# class_names = ['COVID', 'Normal', 'Pneumonia']
# IMG_SIZE = (224, 224)
# DISPLAY_SIZE = (300, 300)

# # -------------------------------
# # Sidebar
# # -------------------------------
# with st.sidebar:
#     st.header("⚙️ Settings")
#     threshold = st.slider("Grad-CAM Sensitivity", 100, 255, 180)

# # -------------------------------
# # Preprocess
# # -------------------------------
# def preprocess_image(img):
#     img = img.resize(IMG_SIZE)
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # -------------------------------
# # Prediction
# # -------------------------------
# def predict_image(img):
#     img_array = preprocess_image(img)
#     predictions = model.predict(img_array)

#     predicted_index = np.argmax(predictions[0])
#     predicted_class = class_names[predicted_index]
#     confidence = predictions[0][predicted_index] * 100

#     return predicted_class, confidence, img_array, predictions[0]

# # -------------------------------
# # Grad-CAM
# # -------------------------------
# def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

#     grad_model = tf.keras.models.Model(
#         inputs=model.input,
#         outputs=[
#             model.get_layer(last_conv_layer_name).output,
#             model.output
#         ]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)

#         if isinstance(predictions, list):
#             predictions = predictions[0]

#         pred_index = tf.argmax(predictions[0])
#         class_channel = predictions[:, pred_index]

#     grads = tape.gradient(class_channel, conv_outputs)

#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]

#     heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
#     heatmap = tf.maximum(heatmap, 0)

#     if tf.reduce_max(heatmap) != 0:
#         heatmap /= tf.reduce_max(heatmap)

#     return heatmap.numpy()

# # -------------------------------
# # Header
# # -------------------------------
# st.markdown("<h1 style='text-align: center;'>🩺 AI Chest X-ray Analyzer</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center;'>COVID-19 | Pneumonia | Normal Detection with Explainable AI</p>", unsafe_allow_html=True)

# st.warning("⚠️ This tool is for educational purposes only.")

# # -------------------------------
# # Upload
# # -------------------------------
# uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:

#     img = Image.open(uploaded_file).convert("RGB")

#     # Smaller preview
#     st.image(img.resize(DISPLAY_SIZE), caption="Uploaded Image")

#     if st.button("🚀 Analyze Image"):

#         with st.spinner("Analyzing..."):

#             predicted_class, confidence, img_array, raw_preds = predict_image(img)

#             # Prediction Card
#             st.markdown(f"""
#             <div style="
#                 background-color:#1e1e1e;
#                 padding:20px;
#                 border-radius:10px;
#                 text-align:center;
#                 color:white;">
#                 <h2>🧠 {predicted_class}</h2>
#                 <h4>Confidence: {confidence:.2f}%</h4>
#             </div>
#             """, unsafe_allow_html=True)

#             # Tabs
#             tab1, tab2 = st.tabs(["📊 Results", "🔥 Visual Analysis"])

#             # ---------------------------
#             # TAB 1: Results
#             # ---------------------------
#             with tab1:
#                 st.subheader("📊 Class Probabilities")
#                 st.bar_chart({
#                     "COVID": raw_preds[0],
#                     "Normal": raw_preds[1],
#                     "Pneumonia": raw_preds[2]
#                 })

#             # ---------------------------
#             # Grad-CAM
#             # ---------------------------
#             heatmap = make_gradcam_heatmap(
#                 img_array,
#                 model,
#                 last_conv_layer_name="block14_sepconv2_act"
#             )

#             img_np = np.array(img.resize(IMG_SIZE))

#             heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
#             heatmap_resized = np.uint8(255 * heatmap_resized)

#             # Full Heatmap
#             heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
#             overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

#             # High Activation Mask
#             mask = np.zeros_like(heatmap_resized)
#             mask[heatmap_resized > threshold] = 255

#             red_mask = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
#             red_mask[:, :, 2] = mask

#             filtered_overlay = cv2.addWeighted(img_np, 1.0, red_mask, 0.7, 0)

#             # Convert BGR → RGB
#             overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
#             filtered_overlay = cv2.cvtColor(filtered_overlay, cv2.COLOR_BGR2RGB)

#             # ---------------------------
#             # TAB 2: Visual Analysis
#             # ---------------------------
#             with tab2:

#                 st.subheader("🧾 Visual Analysis")

#                 col1, col2, col3 = st.columns([1, 1, 1])

#                 with col1:
#                     st.image(img.resize(IMG_SIZE), caption="Original", use_container_width=True)

#                 with col2:
#                     st.image(overlay, caption="Full Heatmap", use_container_width=True)

#                 with col3:
#                     st.image(filtered_overlay, caption="Affected Regions", use_container_width=True)

#                 st.markdown("### 🎨 Heatmap Legend")
#                 st.markdown("""
# - 🔵 Blue → Low importance  
# - 🟢 Green → Medium importance  
# - 🔴 Red → High importance  
# """)

# # -------------------------------
# # Footer
# # -------------------------------
# st.markdown("---")
# st.markdown("👨‍⚕️ AI-Based Medical Diagnosis System | Final Year Project")


import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import gdown
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import tempfile

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI X-ray Analyzer", layout="wide")

# -------------------------------
# Styling
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_trained_model():
    url = "https://drive.google.com/uc?id=1scDHnDOlIF_kIWa6pTcNGWAocyfY706_"  # <-- paste your link here
    gdown.download(url, "model.h5", quiet=False)
    return load_model("model.h5",compile=False)

model = load_trained_model()

class_names = ['COVID', 'Normal', 'Pneumonia']
IMG_SIZE = (224, 224)
DISPLAY_SIZE = (300, 300)

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Grad-CAM Sensitivity", 100, 255, 180)

# -------------------------------
# Preprocess
# -------------------------------
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# Prediction
# -------------------------------
def predict_image(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    return predicted_class, confidence, img_array, predictions[0]

# -------------------------------
# Grad-CAM
# -------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]

        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    if tf.reduce_max(heatmap) != 0:
        heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()

# -------------------------------
# PDF Generator
# -------------------------------
def generate_pdf(img, overlay, filtered_overlay, predicted_class, confidence):

    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("AI Chest X-ray Diagnosis Report", styles['Title']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Prediction: {predicted_class}", styles['Normal']))
    elements.append(Paragraph(f"Confidence: {confidence:.2f}%", styles['Normal']))
    elements.append(Spacer(1, 20))

    # Save temp images
    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    temp_heatmap = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    temp_filtered = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

    Image.fromarray(img).save(temp_img)
    Image.fromarray(overlay).save(temp_heatmap)
    Image.fromarray(filtered_overlay).save(temp_filtered)

    elements.append(Paragraph("Original Image", styles['Heading2']))
    elements.append(RLImage(temp_img, width=300, height=300))

    elements.append(Paragraph("Grad-CAM Heatmap", styles['Heading2']))
    elements.append(RLImage(temp_heatmap, width=300, height=300))

    elements.append(Paragraph("Affected Regions", styles['Heading2']))
    elements.append(RLImage(temp_filtered, width=300, height=300))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Note: AI-assisted diagnosis. Not a substitute for medical advice.", styles['Italic']))

    doc.build(elements)

    return pdf_path

# -------------------------------
# Header
# -------------------------------
st.markdown("<h1 style='text-align: center;'>🩺 AI Chest X-ray Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>COVID-19 | Pneumonia | Normal Detection with Explainable AI</p>", unsafe_allow_html=True)



# -------------------------------
# Upload
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img.resize(DISPLAY_SIZE), caption="Uploaded Image")

    if st.button("🚀 Analyze Image"):

        with st.spinner("🧠 AI is analyzing the X-ray..."):

            predicted_class, confidence, img_array, raw_preds = predict_image(img)

            # Result Card
            color = "#00FF9C" if predicted_class == "Normal" else "#FF4B4B"

            st.markdown(f"""
            <div style="background-color:#1e1e1e;padding:20px;border-radius:10px;text-align:center;">
                <h2 style="color:{color};">{predicted_class}</h2>
                <h4>Confidence: {confidence:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence))

            # Tabs
            tab1, tab2 = st.tabs(["📊 Results", "🔥 Visual Analysis"])

            with tab1:
                st.bar_chart({
                    "COVID": raw_preds[0],
                    "Normal": raw_preds[1],
                    "Pneumonia": raw_preds[2]
                })

            # Grad-CAM
            heatmap = make_gradcam_heatmap(
                img_array,
                model,
                last_conv_layer_name="block14_sepconv2_act"
            )

            img_np = np.array(img.resize(IMG_SIZE))

            heatmap_resized = cv2.resize(heatmap, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
            heatmap_resized = np.uint8(255 * heatmap_resized)

            heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

            mask = np.zeros_like(heatmap_resized)
            mask[heatmap_resized > threshold] = 255

            red_mask = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
            red_mask[:, :, 2] = mask

            filtered_overlay = cv2.addWeighted(img_np, 1.0, red_mask, 0.7, 0)

            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            filtered_overlay = cv2.cvtColor(filtered_overlay, cv2.COLOR_BGR2RGB)

            with tab2:

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.image(img.resize(IMG_SIZE), caption="Original", use_container_width=True)

                with col2:
                    st.image(overlay, caption="Full Heatmap", use_container_width=True)

                with col3:
                    st.image(filtered_overlay, caption="Affected Regions", use_container_width=True)

                st.markdown("### 🎨 Heatmap Legend")
                st.markdown("""
- 🔵 Blue → Low importance  
- 🟢 Green → Medium importance  
- 🔴 Red → High importance  
""")

            # PDF Download
            pdf_file = generate_pdf(img_np, overlay, filtered_overlay, predicted_class, confidence)

            with open(pdf_file, "rb") as f:
                st.download_button(
                    label="📄 Download Full Report (PDF)",
                    data=f,
                    file_name="AI_Xray_Report.pdf",
                    mime="application/pdf"
                )
    st.warning("⚠️ This tool is for educational purposes only.")
