from pathlib import Path
from random import Random
from shutil import copy2


PASTA_PROJETO = Path(__file__).resolve().parent
ORIGEM = PASTA_PROJETO / "dataset_epi"
DESTINO = PASTA_PROJETO / "dataset_epi_mvp"
SEMENTE = 42

# Tamanho final: 10.000 imagens, preservando a proporção 70/20/10.
# Todas as imagens com goggles ou gloves são mantidas.
CONFIGURACAO_SPLITS = {
    "train": {"total": 7000, "negativas": 1750},
    "val": {"total": 2000, "negativas": 500},
    "test": {"total": 1000, "negativas": 250},
}


def classes_do_label(label: Path) -> set[int]:
    classes: set[int] = set()
    for linha in label.read_text(encoding="utf-8").splitlines():
        partes = linha.strip().split()
        if partes:
            classes.add(int(partes[0]))
    return classes


def criar_dataset() -> None:
    if not ORIGEM.is_dir():
        raise FileNotFoundError(f"Dataset de origem não encontrado: {ORIGEM}")
    if DESTINO.exists():
        raise FileExistsError(
            f"A pasta já existe: {DESTINO}. Renomeie-a ou remova-a antes de recriar o MVP."
        )

    aleatorio = Random(SEMENTE)
    resumo: list[str] = []
    total_caixas = {0: 0, 1: 0, 2: 0}

    for split, limites in CONFIGURACAO_SPLITS.items():
        pasta_labels = ORIGEM / "labels" / split
        pasta_imagens = ORIGEM / "images" / split

        prioritarias: list[Path] = []
        apenas_capacete: list[Path] = []
        negativas: list[Path] = []

        for label in pasta_labels.glob("*.txt"):
            classes = classes_do_label(label)
            if classes & {1, 2}:
                prioritarias.append(label)
            elif 0 in classes:
                apenas_capacete.append(label)
            else:
                negativas.append(label)

        aleatorio.shuffle(prioritarias)
        aleatorio.shuffle(apenas_capacete)
        aleatorio.shuffle(negativas)

        quantidade_negativas = min(limites["negativas"], len(negativas))
        quantidade_capacetes = limites["total"] - len(prioritarias) - quantidade_negativas
        if quantidade_capacetes < 0:
            raise ValueError(
                f"O total definido para {split} é menor que as imagens prioritárias e negativas."
            )
        if quantidade_capacetes > len(apenas_capacete):
            raise ValueError(f"Não há imagens suficientes apenas com capacete em {split}.")

        selecionadas = (
            prioritarias
            + apenas_capacete[:quantidade_capacetes]
            + negativas[:quantidade_negativas]
        )
        aleatorio.shuffle(selecionadas)

        novas_imagens = DESTINO / "images" / split
        novos_labels = DESTINO / "labels" / split
        novas_imagens.mkdir(parents=True, exist_ok=True)
        novos_labels.mkdir(parents=True, exist_ok=True)

        imagens_por_stem = {
            imagem.stem: imagem for imagem in pasta_imagens.iterdir() if imagem.is_file()
        }

        for label in selecionadas:
            imagem = imagens_por_stem.get(label.stem)
            if imagem is None:
                raise FileNotFoundError(f"Imagem correspondente não encontrada para {label}")

            copy2(imagem, novas_imagens / imagem.name)
            copy2(label, novos_labels / label.name)

            for linha in label.read_text(encoding="utf-8").splitlines():
                if linha.strip():
                    total_caixas[int(linha.split()[0])] += 1

        resumo.append(
            f"{split}: total={len(selecionadas)}, "
            f"goggles/gloves={len(prioritarias)}, "
            f"apenas_helmet={quantidade_capacetes}, negativas={quantidade_negativas}"
        )

    data_yaml = f"""path: {DESTINO.as_posix()}

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

    resumo.extend(
        [
            f"helmet (0): {total_caixas[0]} caixas",
            f"goggles (1): {total_caixas[1]} caixas",
            f"gloves (2): {total_caixas[2]} caixas",
        ]
    )
    texto_resumo = "\n".join(resumo) + "\n"
    (DESTINO / "resumo.txt").write_text(texto_resumo, encoding="utf-8")
    print(texto_resumo)
    print(f"Dataset MVP criado em: {DESTINO}")


if __name__ == "__main__":
    criar_dataset()
