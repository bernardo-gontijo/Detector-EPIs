import json
from pathlib import Path

import torch
from ultralytics import YOLO


PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_DATASET = PASTA_PROJETO / "dataset_epi_mvp" / "data.yaml"
ARQUIVO_MODELO = (
    PASTA_PROJETO / "runs_epi" / "yolo11n_epi_mvp" / "weights" / "best.pt"
)
PASTA_RESULTADOS = PASTA_PROJETO / "runs_epi"
NOME_AVALIACAO = "avaliacao_final_test"

IMGSZ = 640
BATCH = 16


def verificar_arquivos() -> None:
    if not ARQUIVO_DATASET.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {ARQUIVO_DATASET}")

    if not ARQUIVO_MODELO.is_file():
        raise FileNotFoundError(f"Modelo treinado não encontrado: {ARQUIVO_MODELO}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A GPU não está disponível no PyTorch. Verifique a instalação CUDA "
            "antes de iniciar a avaliação."
        )


def avaliar() -> None:
    verificar_arquivos()

    print("=" * 60)
    print("AVALIAÇÃO FINAL DO MODELO NO CONJUNTO TEST")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Modelo: {ARQUIVO_MODELO}")
    print(f"Dataset: {ARQUIVO_DATASET}")
    print(f"Resolução: {IMGSZ}")
    print(f"Batch: {BATCH}")
    print("=" * 60)

    modelo = YOLO(str(ARQUIVO_MODELO))
    resultados = modelo.val(
        data=str(ARQUIVO_DATASET),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        workers=0,
        plots=True,
        project=str(PASTA_RESULTADOS),
        name=NOME_AVALIACAO,
        exist_ok=False,
    )

    pasta_saida = Path(resultados.save_dir)
    resumo = {
        "split": "test",
        "modelo": str(ARQUIVO_MODELO),
        "dataset": str(ARQUIVO_DATASET),
        "precision": float(resultados.box.mp),
        "recall": float(resultados.box.mr),
        "map50": float(resultados.box.map50),
        "map50_95": float(resultados.box.map),
        "classes": {},
    }

    for indice, nome in modelo.names.items():
        resumo["classes"][nome] = {
            "ap50": float(resultados.box.ap50[indice]),
            "map50_95": float(resultados.box.maps[indice]),
        }

    arquivo_resumo = pasta_saida / "metricas_teste.json"
    arquivo_resumo.write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nAvaliação concluída.")
    print(f"Precisão: {resumo['precision']:.4f}")
    print(f"Recall: {resumo['recall']:.4f}")
    print(f"mAP@50: {resumo['map50']:.4f}")
    print(f"mAP@50-95: {resumo['map50_95']:.4f}")
    print(f"Resultados: {pasta_saida}")
    print(f"Resumo JSON: {arquivo_resumo}")


if __name__ == "__main__":
    avaliar()
