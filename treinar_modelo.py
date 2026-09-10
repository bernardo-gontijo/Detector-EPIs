from pathlib import Path

import torch
from ultralytics import YOLO


# Use "teste" primeiro. Depois, troque para "completo".
MODO = "teste"

PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_DATASET = PASTA_PROJETO / "dataset_epi_mvp" / "data.yaml"
PASTA_RESULTADOS = PASTA_PROJETO / "runs_epi"

CONFIGURACOES = {
    "teste": {
        "model": "yolo11n.pt",
        "epochs": 10,
        "imgsz": 640,
        "batch": 16,
        "patience": 5,
        "name": "teste_mvp",
    },
    "completo": {
        "model": "yolo11n.pt",
        "epochs": 30,
        "imgsz": 640,
        "batch": 16,
        "patience": 8,
        "name": "yolo11n_epi_mvp",
    },
}


def verificar_ambiente() -> None:
    if MODO not in CONFIGURACOES:
        raise ValueError('MODO deve ser "teste" ou "completo".')

    if not ARQUIVO_DATASET.is_file():
        raise FileNotFoundError(f"Arquivo do dataset não encontrado: {ARQUIVO_DATASET}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A GPU não está disponível no PyTorch. A instalação atual provavelmente "
            "é somente para CPU. Configure a versão CUDA do PyTorch antes de iniciar "
            "o treinamento completo."
        )


def treinar() -> None:
    verificar_ambiente()
    config = CONFIGURACOES[MODO]

    print("=" * 60)
    print("TREINAMENTO DE DETECÇÃO DE EPIs")
    print(f"Modo: {MODO}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Modelo: {config['model']}")
    print(f"Épocas: {config['epochs']}")
    print(f"Dataset: {ARQUIVO_DATASET}")
    print(f"Resultados: {PASTA_RESULTADOS / config['name']}")
    print("=" * 60)

    modelo = YOLO(config["model"])
    modelo.train(
        data=str(ARQUIVO_DATASET),
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=0,
        workers=0,
        cache=False,
        patience=config["patience"],
        project=str(PASTA_RESULTADOS),
        name=config["name"],
        exist_ok=False,
        plots=True,
        seed=42,
    )

    print("\nTreinamento concluído.")
    print(f"Confira os resultados em: {PASTA_RESULTADOS / config['name']}")


if __name__ == "__main__":
    treinar()
