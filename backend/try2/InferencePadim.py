import matplotlib.pyplot as plt
from anomalib.data.utils import read_image
from pathlib import Path
from anomalib.deploy import OpenVINOInferencer
from anomalib.visualization import ImageVisualizer


dataset_path = Path.cwd()/"WConn"
weights_path = Path.cwd()/"weights"

image_path = r"C:/Users/pega_user/Desktop/aoi-inspector/backend/try2/P16_0_20251031_182620.jpg"
image = read_image(path=r"C:/Users/pega_user/Desktop/aoi-inspector/backend/try2/P16_0_20251031_182620.jpg")
plt.imshow(image)

openvino_model_path = weights_path/"openvino"/"model.bin"
metadata_path = weights_path/"openvino"/"metadata.json"

print("OpenVino model exits:", openvino_model_path.exists())
print("OpenVino path", openvino_model_path)
print("Metadata model exits", metadata_path.exists())
print("Metadata path", metadata_path)

inferencer = OpenVINOInferencer(
    path=openvino_model_path,
    device="CPU",
)
print(image.shape)
predictions= inferencer.predict(image=image)

visualizer = ImageVisualizer()
output_image = visualizer.visualize(predictions)
plt.imshow(output_image)
print("imagen visualizada")