from anomalib.deploy import OpenVINOInferencer
from anomalib.utils.visualization.image import ImageVisualizer
import numpy as np
import matplotlib.pyplot as plt

model = OpenVINOInferencer(path="model.xml", task="classification")
predictions = model.predict(image="P16_0.jpg")
print("predictions", predictions)

#visualizer = ImageVisualizer()
#vis_result = visualizer.visualizee(predictions)

plt.imshow(predictions)
plt.axis("off")
plt.show()