#train_all_models.py
import os
os.environ['ANOMALIB_DISABLE_SYMLINKS'] = '1'
import torch, gc
torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')
from pathlib import Path
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.deploy import ExportType
from datetime import datetime
import json
from lightning.pytorch.callbacks import TQDMProgressBar

def train_component(component_name, export_onnx=False, export_OV=True):

    print(f"\n{'='*60}")
    print(f"Entrenando modelo para: {component_name}")
    print(f"\n{'='*60}\n")

    datamodule = Folder(
        name= component_name, 
        root= f"./dataset_conn_anomalib/{component_name}",
        normal_dir= "train/good",
        #abnormal_dir= "test/defect", # no se usa para clasificacion solo en segmentacion con mascaras
        normal_test_dir= "test/defect",
        #test_split_mode="from_dir",
        task= "classification",
        image_size=(224,224),
        train_batch_size= 4,
        eval_batch_size= 4,
        num_workers=0 #ultimo cambio sin probar aun
    )
    model = Patchcore(
        #backbone= "wide_resnet50_2", #ultimo cambio sin ptobar
        #layers= ["layer2", "layer3"],
        num_neighbors= 9
    )
    engine = Engine(
        task="classification",
        #normalization= "min_max",
        #threshold= "F1AdaptiveThreshold",
        image_metrics=["F1Score", "AUROC"],
        pixel_metrics=["F1Score", "AUROC"],
        accelerator = "gpu",
        devices = 1,
        precision="16-mixed",
        max_epochs=10,
        default_root_dir=f".results/{component_name}",
        #enable_progress_bar=False,
        #callbacks=[TQDMProgressBar(refresh_rate=10)],
    )
    start_time = datetime.now()
    print(f"Inicio{start_time.strftime('%H:%M:%S')}")
    engine.fit(model=model, datamodule=datamodule)
    training_time = (datetime.now()-start_time).total_seconds()
    print(f"Tiempo de entrenamiento {training_time:.2f} segundos")

    ckpt_path = f".results/{component_name}/Patchcore/{component_name}/v0/weights/lightning/model.ckpt"
    print(f"Exportando modelo desde checkpoint: {ckpt_path}")

    trained_model = Patchcore.load_from_checkpoint(ckpt_path)

    engine.export(
        model=trained_model,
        export_type=ExportType.TORCH,
        export_root=f"./models/{component_name}"
    )
    print(f"Modelo Torch exportado a ./models/{component_name}/")

    '''if export_onnx==True:
        engine.export(
            model=trained_model,
            export_type=ExportType.ONNX,
            export_root=f"./models/{component_name}_onnx"
        )
        print(f"Modelo ONNX exportado a ./models/{component_name}_onnx/")'''
    if export_OV==True:
        engine.export(
            model=trained_model,
            export_type=ExportType.OPENVINO,
            export_root=f"./models/{component_name}_openvino"
        )
        print(f"Modelo OpenVINO exportado a ./models/{component_name}_openvino/")


    try:

        test_results= engine.test(model=model, datamodule=datamodule)
        results={
            "component": component_name,
            #"image_AUROC": float(test_results[0]['image_AUROC']),
            #"pixel_AUROC": float(test_results[0]['pixel_AUROC']),
            #"image_F1Score": float(test_results[0]["image_F1Score"]),
            #"training_time_seconds": training_time,
            "timestamp": datetime.now().isoformat()
        }

        print(f"\n{'='*60}")
        print(f"Resultados para {component_name}:")
        #print(f"Image AUROC: {results['image_AUROC']:.3f}")
        #print(f"Pixel AUROC: {results['pixel_AUROC']:.3f}")
        #print(f"Image F1 {results['image_F1Score']:.3f}")
        print(f"{'='*60}\n")

    except Exception as e:
        print("Error al terminar el entrenamiento", e)


    '''print("Exportando modelo PyTorch ...")
    engine.export(
        model=model,
        export_type=ExportType.TORCH,
        export_root=f"./models/{component_name}",
    )
    print(f"PyTorch model guardado en ./models/{component_name}/")

    if export_onnx:
        print(f"Exportando modelo ONNX...")
        engine.export(
            model=model,
            export_type=ExportType.ONNX,
            export_root=f"./models/{component_name}_onnx",
        )
        print(f"ONNX model guardado en: ./models/{component_name}_onnx")
    elif export_OV:
        print(f"Exportando modelo OpenVINO...")
        engine.export(
            model=model,
            export_type=ExportType.OPENVINO,
            export_root=f"./models/{component_name}_openvino",
        )
        print(f"OpenVINO model guardado en: ./models/{component_name}_openvino")'''
    return results

def train_all_components(components_list, export_onxx=False, export_OV=True):
    
    all_results = []
    failed = []

    print(f"\n{'='*60}")
    print(f"Entrenando modelo para: {len(components_list)}")
    print(f"\n{'='*60}\n")

    for i, component in enumerate(components_list, 1):
        try:
            print(f"\n[{i}/{len(components_list)}] Procesando {component}")
            result = train_component(component, export_onnx=export_onxx, export_OV=export_OV)
            all_results.append(result)
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print("Error en", component, e)
            failed.append(component)
            continue
    
    summary = {
        "total_components": len(components_list),
        "successful": len(all_results),
        "failed": failed,
        "results": all_results,
        "timestamp": datetime.now().isoformat()
    }

    summary_path = "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    
    print(f"\n" + "="*60)
    print("Resumen final")
    print("="*60)
    print(f"Exitosos: {len(all_results)}/{len(components_list)}")
    if failed:
        print(f"Fallidos: {len(failed)}-{failed}")
    print(f"\nPromedio Image AUROC: {sum(r['image_AUROC'] for r in all_results)/len(all_results):.3f}")
    print(f"Resumen guardado en: {summary_path}")
    print(f"{'='*60}\n")

    return summary

if __name__ == "__main__":
    components = ["OrConn1", "OrConn2", "WConn"]
    summary = train_all_components(components, export_onxx=False, export_OV=True)