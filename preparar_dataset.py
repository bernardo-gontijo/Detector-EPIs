from pathlib import Path
from shutil import copy2


ORIGEM = Path(__file__).resolve().parent / "datasets_original"
DESTINO = Path(__file__).resolve().parent / "dataset_epi"

# Índice original -> índice final
MAPA_CLASSES = {
    3: 0,  # Hardhat -> helmet
    2: 1,  # Goggles -> goggles
    1: 2,  # Gloves -> gloves
}

SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def converter() -> None:
    if not ORIGEM.is_dir():
        raise FileNotFoundError(f"Dataset original não encontrado: {ORIGEM}")

    if DESTINO.exists():
        raise FileExistsError(
            f"A pasta de destino já existe: {DESTINO}. "
            "Remova-a ou renomeie-a antes de executar novamente."
        )

    totais_classes = {0: 0, 1: 0, 2: 0}
    total_imagens = 0
    total_negativas = 0

    for split_origem, split_destino in SPLITS.items():
        pasta_imagens = ORIGEM / split_origem / "images"
        pasta_labels = ORIGEM / split_origem / "labels"

        novas_imagens = DESTINO / "images" / split_destino
        novos_labels = DESTINO / "labels" / split_destino
        novas_imagens.mkdir(parents=True, exist_ok=True)
        novos_labels.mkdir(parents=True, exist_ok=True)

        for imagem in pasta_imagens.iterdir():
            if not imagem.is_file() or imagem.suffix.lower() not in EXTENSOES_IMAGEM:
                continue

            copy2(imagem, novas_imagens / imagem.name)
            total_imagens += 1

            label_original = pasta_labels / f"{imagem.stem}.txt"
            novo_label = novos_labels / f"{imagem.stem}.txt"
            novas_linhas: list[str] = []

            if label_original.exists():
                for numero_linha, linha in enumerate(
                    label_original.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    partes = linha.strip().split()
                    if not partes:
                        continue
                    if len(partes) != 5:
                        raise ValueError(
                            f"Anotação inválida em {label_original}, linha {numero_linha}: {linha}"
                        )

                    classe_antiga = int(partes[0])
                    if classe_antiga not in MAPA_CLASSES:
                        continue

                    classe_nova = MAPA_CLASSES[classe_antiga]
                    partes[0] = str(classe_nova)
                    novas_linhas.append(" ".join(partes))
                    totais_classes[classe_nova] += 1

            if not novas_linhas:
                total_negativas += 1

            conteudo = "\n".join(novas_linhas)
            if conteudo:
                conteudo += "\n"
            novo_label.write_text(conteudo, encoding="utf-8")

    data_yaml = """path: .

train: images/train
val: images/val
test: images/test

nc: 3

names:
  0: helmet
  1: goggles
  2: gloves
"""
    (DESTINO / "data.yaml").write_text(data_yaml, encoding="utf-8")

    print(f"Dataset criado em: {DESTINO}")
    print(f"Imagens: {total_imagens}")
    print(f"Imagens sem as três classes: {total_negativas}")
    print(f"helmet (0): {totais_classes[0]} caixas")
    print(f"goggles (1): {totais_classes[1]} caixas")
    print(f"gloves (2): {totais_classes[2]} caixas")


if __name__ == "__main__":
    converter()
