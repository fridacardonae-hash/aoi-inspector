import cv2
import numpy as np
from anomalib.deploy import OpenVINOInferencer
from pathlib import Path
from PIL import Image, ImageTk
from anomalib.visualization import ImageVisualizer
import json
import cv2
import customtkinter as ctk
import os

#Crear interfaz
root = ctk.CTk()
root.geometry("800x600")
root.title("Resultado de Inferencia")


#Abrir configuracion
with open("config.json", "r") as f:
    file_config = json.load(f)

componentes = file_config["Components"]
print("componentes encontrados para inspeccionar:", componentes)

#Cargar modelo OpenVINO
weights_path = Path.cwd()/"weights"
openvino_model_path = weights_path/"openvino" /"model.bin"

print("OpenVino model exits:", openvino_model_path.exists())
print("OpenVino path", openvino_model_path)

#Abrir imagen a hacer inferencia
#imagen_path=r"C:\Users\pega_user\Desktop\aoi-inspector\backend\try2\WConn\train\defect\P16_0_20251110_174532.jpg" #min NG
#imagen_path=r"C:/Users/pega_user/Desktop/aoi-inspector/backend/try2/WConn/train/defect/P16_0_20251110_175449.jpg" #max NG
imagen_path=r"C:/Users/pega_user/Desktop/aoi-inspector/backend/try2/WConn/train/good/P16_0_20251031_182610.jpg" #OK
image = Image.open(imagen_path)


#hacer inferencia
inferencer = OpenVINOInferencer(
    path=openvino_model_path,
    device="CPU",
)
#obtener predicciones
predictions= inferencer.predict(image=image)
score = predictions.pred_score
print("prediction", score)


#Guardar solo imagen de anomalia
vizualizer2 = ImageVisualizer(
    fields=[""],
    overlay_fields=[("image",["anomaly_map"])],
    text_config={"size":14},

)
output_anomalyimg = vizualizer2.visualize(predictions)
output_anomalyimg.save("output_anomaly.png")

#Visualizar todas los modos de la inferencia 
visualizer = ImageVisualizer()
output_image = visualizer.visualize(predictions)

#cargar areas de recorte de roi
area = file_config["ROIs"]
x1 = area["x1"]
y1 = area["y1"]
x2 = area["x2"]
y2 = area["y2"]

#Recorte con OpenCV [y1:y2, x1:x2]
imroi = cv2.imread("output_anomaly.png")
roi= imroi[y1:y2, x1:x2]
cv2.imshow("roi", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Analisis con openCV en el ROI 
mean_color = cv2.mean(roi)[:3]
b_mean, g_mean, r_mean = mean_color
print(f"Promedio BGR: {mean_color}")
cv2.imshow("roi", roi)

#Mostrar set de imagenes
img_tk = ImageTk.PhotoImage(output_image)
label = ctk.CTkLabel(root, image=img_tk, text="")
label.pack(padx=20, pady=20, expand=True)
label.image = img_tk
root.mainloop()

#cerrar roi
cv2.waitKey(0)
cv2.destroyAllWindows()
