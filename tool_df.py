import shutil
import pandas as pd
import datetime as datetime
import json
import os
import time

OK_folder = r"C:\Users\pega_user\Desktop\aoi-inspector\backend\tools_df\OK"
NG_folder = r"C:\Users\pega_user\Desktop\aoi-inspector\backend\tools_df\NG"
samples_folderok = r"\\10.232.4.39\pmxbushare\PMX-BU8-ENG\17. Set-up Team\Frida\Nueva carpeta\11_24"


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
for key, value in folder_comp.items():
    img_name = key
    folder_name = value
    for root, dirs, files in os.walk(samples_folderok):
        for f in files:
           
            if f == f"{img_name}.jpg":
                print("file", f)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img = f"{img_name}_{timestamp}.jpg"
                origen = os.path.join(root, f)
                destino = os.path.join(OK_folder, folder_name,img)
                print("origen", origen)
                print("destino", destino)
                print("copiando")
                shutil.copy(origen, destino)
                time.sleep(1)





    
