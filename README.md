# Detecção de EPIs com YOLO11

Projeto de visão computacional para detectar três tipos de Equipamentos de Proteção Individual (EPIs) em imagens:

- `helmet`: capacete;
- `goggles`: óculos de proteção;
- `gloves`: luvas.

O projeto realiza **detecção de objetos**: o modelo informa a classe, a confiança e a caixa delimitadora de cada EPI encontrado. Nesta etapa, ele não verifica se o equipamento está sendo usado corretamente por uma pessoa.

## Estado do projeto

O pipeline atual inclui:

- conversão do dataset original de 14 para 3 classes;
- criação de um subconjunto balanceado para MVP;
- treinamento com YOLO11;
- validação durante o treinamento;
- geração de curvas, matrizes de confusão e pesos do modelo.

Baseline de validação obtido com YOLO11n após 10 épocas:

| Métrica | Resultado |
|---|---:|
| Precisão | 78,0% |
| Recall | 84,9% |
| mAP@50 | 84,2% |
| mAP@50–95 | 45,0% |

Resultados por classe em AP@50:

| Classe | AP@50 |
|---|---:|
| Helmet | 62,1% |
| Goggles | 96,6% |
| Gloves | 94,0% |

Esses valores foram obtidos no conjunto de **validação**, não no conjunto de teste final. Eles não representam desempenho garantido em câmeras ou ambientes diferentes.

## Fontes dos dados

O dataset processado pelos scripts e utilizado no baseline atual veio de:

- [Personal Protective Equipment — Combined Model, Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model/), licença CC BY 4.0;

O [Construction Site Safety Image Dataset, Kaggle](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow/data), também sob licença CC BY 4.0, foi avaliado como possível fonte adicional para capacetes, mas não é incorporado automaticamente pelos scripts atuais.

Os datasets não são armazenados neste repositório. Consulte as páginas originais para download, atribuição e condições de uso.

## Estrutura do repositório

```text
Projeto_EPI/
├── configs/
│   └── data.example.yaml
├── avaliar_modelo.py
├── criar_dataset_mvp.py
├── iniciar_avaliacao.bat
├── iniciar_treinamento.bat
├── preparar_dataset.py
├── requirements.txt
├── treinar_modelo.py
└── README.md
```

Pastas criadas localmente durante o uso, mas não enviadas ao GitHub:

```text
datasets_original/   # download original, sem modificações
dataset_epi/         # dataset convertido para três classes
dataset_epi_mvp/     # subconjunto balanceado de 10 mil imagens
runs_epi/            # métricas, gráficos e pesos dos treinamentos
ultralytics/         # ambiente virtual de treinamento
label-studio-py312/  # ambiente virtual opcional do Label Studio
```

## Pré-requisitos

- Windows 10 ou 11;
- Python 3.12 recomendado;
- Git;
- GPU NVIDIA recomendada para treinamento;
- driver NVIDIA atualizado;
- pelo menos 8 GB de RAM, preferencialmente 16 GB ou mais.

O Label Studio é opcional e só é necessário para revisar ou criar anotações.

## Instalação

Clone o repositório:

```powershell
git clone URL_DO_REPOSITORIO
cd Projeto_EPI
```

Crie o ambiente virtual com o nome esperado pelo iniciador do Windows:

```powershell
py -3.12 -m venv ultralytics
```

Ative o ambiente:

```powershell
.\ultralytics\Scripts\Activate.ps1
```

### Instalação com GPU NVIDIA

Instale primeiro o PyTorch compatível com seu sistema. Consulte o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/) e escolha Windows, Pip, Python e a plataforma CUDA indicada.

Exemplo utilizado durante o desenvolvimento com CUDA 13.0:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Instale as demais dependências:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirme que a GPU está disponível:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU não detectada')"
```

O resultado esperado para a primeira linha é `True`.

## Preparação do dataset

### 1. Download

Baixe o dataset do Roboflow no formato **YOLOv11**, usando a opção **Download zip to computer**.

Extraia o conteúdo para:

```text
datasets_original/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

### 2. Conversão das classes

Execute:

```powershell
python preparar_dataset.py
```

O script preserva o dataset original e cria `dataset_epi/` com o seguinte mapeamento:

| Classe original | Índice original | Classe final | Índice final |
|---|---:|---|---:|
| Hardhat | 3 | helmet | 0 |
| Goggles | 2 | goggles | 1 |
| Gloves | 1 | gloves | 2 |

As demais classes são removidas das anotações. Imagens sem as três classes recebem arquivos de anotação vazios e funcionam como exemplos negativos.

O script interrompe se a pasta `dataset_epi/` já existir. Renomeie ou remova conscientemente a versão anterior antes de recriá-la.

### 3. Criação do MVP balanceado

Execute:

```powershell
python criar_dataset_mvp.py
```

O script cria `dataset_epi_mvp/` com 10 mil imagens:

| Divisão | Imagens | Finalidade |
|---|---:|---|
| Train | 7.000 | Atualizar os pesos do modelo |
| Val | 2.000 | Selecionar o melhor modelo e acompanhar métricas |
| Test | 1.000 | Avaliação final independente |

Todas as imagens com óculos ou luvas são priorizadas. A quantidade de imagens somente com capacete e de imagens negativas é limitada para reduzir o desequilíbrio.

O arquivo `configs/data.example.yaml` mostra a estrutura esperada. O `data.yaml` real é criado automaticamente dentro do dataset e contém um caminho local, portanto não é versionado.

## Treinamento

As configurações ficam no início de `treinar_modelo.py`.

Para um teste curto, use:

```python
MODO = "teste"
```

Para o treinamento principal, use:

```python
MODO = "completo"
```

No Windows, inicie com duplo clique em:

```text
iniciar_treinamento.bat
```

Também é possível iniciar pelo terminal:

```powershell
python treinar_modelo.py
```

Configuração principal atual:

```text
Modelo: yolo11n.pt
Épocas: 30
Resolução: 640 × 640
Batch: 16
Patience: 8
GPU: device 0
Workers: 0
```

Se ocorrer falta de memória da GPU, reduza `batch` de `16` para `8`. Em Windows, `workers=0` é utilizado para evitar consumo excessivo de RAM por processos paralelos.

## Resultados do treinamento

Cada execução cria uma pasta dentro de `runs_epi/`. Os arquivos mais importantes são:

```text
weights/best.pt                 # melhor modelo segundo a validação
weights/last.pt                 # estado da última época
results.csv                     # métricas por época
results.png                     # evolução das perdas e métricas
confusion_matrix.png            # matriz de confusão em valores absolutos
confusion_matrix_normalized.png # matriz de confusão normalizada
BoxPR_curve.png                 # curva Precision–Recall
BoxF1_curve.png                 # F1 por limiar de confiança
```

Durante o desenvolvimento, prefira `best.pt` para validação, teste e inferência.

## Avaliação no conjunto de teste

O conjunto `test` não é utilizado durante o treinamento. Avalie-o somente depois de escolher o modelo e as configurações finais:

No Windows, execute com duplo clique:

```text
iniciar_avaliacao.bat
```

Também é possível iniciar pelo terminal:

```powershell
python avaliar_modelo.py
```

O script usa `runs_epi/yolo11n_epi_mvp/weights/best.pt`, avalia exclusivamente o split `test` e salva gráficos, matriz de confusão e `metricas_teste.json` dentro de `runs_epi/avaliacao_final_test/`.

Não reajuste repetidamente o modelo com base no resultado de teste. Caso isso aconteça, o conjunto deixa de representar uma avaliação final imparcial.

## Inferência em imagens

Para detectar EPIs em uma imagem:

```powershell
yolo detect predict model="runs_epi/yolo11n_epi_mvp/weights/best.pt" source="caminho/para/imagem.jpg" conf=0.35 device=0 save=True
```

Para processar uma pasta:

```powershell
yolo detect predict model="runs_epi/yolo11n_epi_mvp/weights/best.pt" source="caminho/para/imagens" conf=0.35 device=0 save=True
```

O valor `conf=0.35` é um ponto inicial. Um valor menor aumenta a quantidade de detecções e falsos positivos; um valor maior reduz falsos positivos, mas pode deixar EPIs sem detecção.

## Interpretação resumida das métricas

- **Precisão:** entre as detecções produzidas, quantas estão corretas;
- **Recall:** entre os objetos reais, quantos foram encontrados;
- **mAP@50:** qualidade média das classes com sobreposição mínima de 50%;
- **mAP@50–95:** avaliação mais rigorosa da classificação e do posicionamento das caixas;
- **Background na matriz:** previsões sem correspondência com uma anotação ou objetos reais que não receberam uma previsão.

## Limitações atuais

- O resultado de capacetes é inferior ao de óculos e luvas;
- objetos pequenos ou distantes continuam difíceis;
- o dataset pode conter imagens semelhantes entre as divisões;
- o desempenho de validação pode não representar ambientes reais;
- a associação entre EPI e pessoa ainda não foi implementada.

