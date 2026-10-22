# aoi-inspector

Proyecto experimental de inspección óptica con interfaz de escritorio y varias versiones de entrenamiento e inferencia Anomalib.

## Alcance de esta documentación

Basada en la revisión estática de los archivos locales del 22 de octubre de 2026. Los comandos son instrucciones de arranque deducidas del código; no se ejecutaron aplicaciones, instalaciones, entrenamientos ni conexiones a equipos.

## Estructura

| Ruta | Contenido |
| --- | --- |
| `backend/` | Lógica de inspección, procesamiento o servicios. |
| `frontend/` | Interfaz de usuario y recursos de presentación. |

## Preparación

Se necesita Python y un entorno compatible con las versiones declaradas. La versión exacta de Python no se ha validado. En PowerShell, desde esta carpeta:

```powershell
python -m venv .venv-doc
.\.venv-doc\Scripts\Activate.ps1
python -m pip install -r "frontend/requirements.txt"
```

## Configuración

Revisar `frontend/config.json` antes del arranque. Sus claves de primer nivel son:

`Components`, `Image_correlation`, `ROIs`, `Modelbinpath`, `enviados_log`, `scorelimit`, `log_path`, `model v3`, `PLC_IP`, `PLC_PORT`.

Ajustar rutas de imágenes y modelos, regiones de interés y límites de clasificación según la estación. Si aparecen PLC_IP y PLC_PORT, configurarlos para el equipo correspondiente. Mantener los archivos de modelo y metadatos requeridos por el código. Las rutas locales o compartidas deben existir y ser accesibles.

## Ejecución

Desde la raíz de esta carpeta, con el entorno activado:

```powershell
Set-Location frontend
python main.py
```

## Flujo de trabajo

1. Revisar configuración, modelos y carpetas de entrada/salida.
2. Abrir la interfaz y cargar las imágenes o activar el modo de seguimiento disponible.
3. Consultar los resultados de inspección y los registros generados por la variante utilizada.
4. Validar con muestras conocidas antes de habilitar comunicación con el PLC. Los módulos PLC incluyen escritura de bits.

## Observaciones específicas

backend contiene try2, try3 y una variante histórica cuyo nombre indica problemas con Anomalib 1.1. No mezclar sus dependencias sin revisar los requirements.txt de cada experimento.

## Verificación y mantenimiento

La documentación se comprobó contra la estructura y el código local, sin prueba funcional. Si falla un import, revisar el entorno y el manifiesto de la variante; si falla una lectura, comprobar el directorio de trabajo y las rutas configuradas. Actualizar este README cuando cambien entradas, dependencias o configuración.

---

## Documentación adicional conservada de GitHub

# AOI Inspector - Anomalib 

Main issue was current AOI in production line can’t detect an specific failure mode due to the variability on the position and the size of the failure.
Main branch is dedicated for the UI implemented to obtain the inference of different pictures saved in a specific folder path with a custom trained nomalib model (pipeline used is in the modeltrain branch).

The UI includes two tabs one for “manual” inference, where all the pictures stored in the folder selected will be analized by the model then youll be able to navigate between all the pictures with the arrows or the “left” “right” keys on the keyboard.

The second tab is the “Online Inspection” which based on the configuration of the “Config2.json” will constantly look for a new folder in the path selected once there’s a new folder will look for the specific name of a picture and perform the snip of the ROI and make the inference.

This method is very useful to install this inference app in a machine like an AOI and perform an extra inspection with the pictures stored.

Points to consider:
Current process flow for both manual and online  inspection performs a snip on the original picture for an specific ROI which extracts exact the same region of the picture used to train the model (this was due to specific reasons of this application lease check the modeltrain branch for more info)


Df_tools branch

This three scripts are the first ones I used to generate, segregate and clean my data in order to have a well organized, and clean dataset.
1.- Dftools1
This scripts extracts specifically the pictures named as the config.json file describes, from a specified path where different folders contain many other pictures (im only interested in create a model for a specific component)
Once the pictures are extracted it copies them into another folder.
2.-Segregator
Then I use the “Segregator.py”
Which is almost same UI than  the used in the main branch but in this case I use it to clean my data and prepare it for next step with this tool I can visualize only the ROI im interested to decide if is a good or defect and make sure my data for defect will have only the failure modes im expecting to detect with my model
By clicking in the OK or NG button it will separate the pictures into different folders.

3.-Df tools2.-
This tool is the one I use once I have separated my pictures OK from NG to snip the ROI im interested and have my final data set.
I did this one because previously I trained a model with the full picture, but the variations of external components to the ones im interested were much and I had different values of anomaly score, and also because the failure mode im willing to detect as an anomaly is kinda small and only occurs inside the ROI.

Anomalibpadim branch
This is just the pipeline I used to train my model with Anomalib framework using PaDiM model with the backbone Resnet18 
I initially tried a to train the model usin Patchcore but required much more configuration, so I decided to use PaDiM to have a quicker solution, of course the quality of the model depends a lot on the dataset, so after a few clean ups and different tries I got an Image_AUROC of 0.89 and and F1Score of 0.65 (I know stills need improvement)
