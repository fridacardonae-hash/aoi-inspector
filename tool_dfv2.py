import shutil
import pandas as pd
import datetime as datetime
import json
import os
import time
import cv2

OK_folder = r"C:\Users\pega_user\Desktop\aoi-inspector\backend\tools_df\OK_snip"
NG_folder = r"C:\Users\pega_user\Desktop\aoi-inspector\backend\tools_df\NG_snip"
#samples_folderok = r"C:\Users\pega_user\Desktop\aoi-inspector\backend\tools_df\defect"
samples_folderok = r"C:\Users\pega_user\Desktop\aoi-inspector\backend\tools_df\OK\WConn\OK"


i=0
with open ("COMPONENS.json", "r") as f:
    folder_comp = json.load(f)

#Crear folders por componente
for key, value in folder_comp.items():
    img_name = key
    folder_name = value
    #print(f"folder {folder_name}, img {img_name}")
    folder_pathok = os.path.join(OK_folder,folder_name)
    folder_pathng = os.path.join(NG_folder, folder_name)
    os.makedirs(folder_pathok, exist_ok=True)
    os.makedirs(folder_pathng, exist_ok=True)
    print("Folders listos")

#separar imagenes por folder

for root, dirs, files in os.walk(samples_folderok):
        for f in files:
            print("f", f)
            img_orig = os.path.join(samples_folderok, f)
            print("file", img_orig)
            
            #Recorte con OpenCV [y1:y2, x1:x2]
            imroi = cv2.imread(img_orig)
            roi= imroi[1200:2433, 1085:3894]
            #cv2.imshow("roi", roi)
            
            #roi2 = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) 
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            img = f"{f}_{timestamp}.jpg"
            destino = os.path.join(folder_pathok,img)

            cv2.imwrite(destino, roi)

            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            time.sleep(1)





    
