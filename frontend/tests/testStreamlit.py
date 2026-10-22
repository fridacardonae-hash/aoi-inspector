import streamlit as st
import cv2
import numpy as np
from anomalib.deploy import OpenVINOInferencer
from pathlib import Path
from PIL import Image
from anomalib.visualization import ImageVisualizer

weights_path = Path.cwd()/"weights"
openvino_model_path = weights_path/"openvino"/"model.bin"

st.title("🔍 Anomalib Inspector")
st.write("Visualizador de inferencias con heatmap por componente")

# Carga del modelo
#componentes = [p.name for p in weights_path.iterdir() if p.is_dir()]
componentes = ["WConn"]
componente = st.selectbox("Selecciona el componente", componentes)

# Selección de imagen
imagen_file = st.file_uploader("Sube una imagen para analizar", type=["jpg","png","jpeg"])
if imagen_file:
    image = np.array(Image.open(imagen_file).convert("RGB"))

    inferencer = OpenVINOInferencer(
    path=openvino_model_path,
    device="CPU",
    )
    predictions= inferencer.predict(image=image)
    score = predictions.pred_score
    print("prediction", score)

    visualizer = ImageVisualizer()
    output_image = visualizer.visualize(predictions)
    output_image

    # Visualizaciones
    st.image(output_image, caption="Output", use_container_width=True)


    st.metric("Probabilidad de anomalía", f"{score}%")
