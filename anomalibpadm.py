from anomalib.data import Folder
from pathlib import Path
from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer, ExportType
from anomalib.models import Padim
from anomalib.engine import Engine


def model_training():
    datamodule = Folder(
        name="WConn",
        root=Path.cwd()/"WConn",
        normal_dir= r"C:/Users/pega_user/Desktop/aoi-inspector/backend/try2/WConn/train/good",
        abnormal_dir=r"C:/Users/pega_user/Desktop/aoi-inspector/backend/try2/WConn/train/defect",
        normal_split_ratio=0.2,
        #image_size=(256,256),
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0
    )

    datamodule.setup()
    print("DATAMODULE SETEADO")

    i, data = next(enumerate(datamodule.val_dataloader()))
    print("Imprimiendo keys")
    print(data.keys())
    print(data["image"].shape)

    model = Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
    )
    engine = Engine(accelerator="cpu", devices=1)
    print("Entrenando modelo Padim")
    engine.fit(model=model, datamodule=datamodule)
    print("Testeando modelo")
    test_results=engine.test(model=model, datamodule=datamodule)
    print("Resultados", test_results)

    print("Exportando modelo")
    openvino_model_path = engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
        export_root=str(Path.cwd()),
        )
    print("Modelo exportado correctamente")


if __name__ == "__main__":
    model_training()