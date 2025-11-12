# 🌱 Guia de Execução — EcoVision Cerrado

Este documento descreve o passo a passo para configurar, treinar e executar o projeto **EcoVision Cerrado**, garantindo reprodutibilidade completa do ambiente e dos resultados.

---

## ⚙️ 1️⃣ Requisitos

Antes de começar, instale:

- **Python 3.11+**
- **Git**
- **pip** (gerenciador de pacotes)
- **Virtualenv** (recomendado)
- **Streamlit** e dependências listadas em `requirements.txt`

---

## 🧩 2️⃣ Clonar o Repositório e Preparar o Ambiente

```bash
git clone https://github.com/HigorPereira10/ecovision-cerrado.git
cd ecovision-cerrado

# Criar e ativar ambiente virtual
python -m venv .venv
.\.venv\Scripts\activate     # Windows
# ou
source .venv/bin/activate    # Linux/macOS

# Instalar dependências
pip install -r requirements.txt
```

## 3️⃣ Opções de Execução

O projeto pode ser executado de duas formas, dependendo do objetivo do usuário.

# 🅰️ Opção 1 — Usar o Dataset EuroSAT (treinar IA do zero)
```bash
Esta opção faz o download, extração e pré-processamento do dataset EuroSAT (base de imagens de satélite multiclasse).
python src/download_dataset.py
python src/extract_eurosat.py
python src/preprocess.py
```
# 📦 Resultado:
Imagens prontas em data/processed/, com 10 classes organizadas em subpastas (ex: Forest, River, Industrial, etc.).

# 🧠 Treinar o modelo de IA

Com os dados processados, treine a IA baseada na arquitetura ResNet18 (PyTorch):
```bash
python src/model/train_model.py
```
# 💾 O modelo treinado será salvo em:

```bash
outputs/model_cerrado_resnet18.pth
```

# 🅱️ Opção 2 — Usar imagens reais do Cerrado (análise multitemporal)

Esta opção analisa o desmatamento e regeneração em áreas reais do Cerrado, comparando imagens de 2020 e 2024.

1️⃣ Baixar imagens no Google Earth Engine

Acesse o Google Earth Engine Code Editor

Use o script sentinel_composite fornecido no repositório

Exporte as imagens .tif (ex: area_001_2020.tif, area_001_2024.tif)

Salve-as em:

```bash
data/raw/area_001/
```

2️⃣ Converter para .jpg

```bash
python scripts/tif_to_jpg_512.py "data/raw/area_001/area_001_2020.tif" "data/multitemporal/area_001/2020.jpg"
python scripts/tif_to_jpg_512.py "data/raw/area_001/area_001_2024.tif" "data/multitemporal/area_001/2024.jpg"
```

3️⃣ Processar e gerar métricas

```bash
python scripts/pair_and_process_multitemporal.py --root data/multitemporal --year1 2020 --year2 2024
```

4️⃣ Visualizar comparações e heatmaps

Os resultados estarão em:

```bash
outputs/plots/
```

## 📊 4️⃣ Executar o Dashboard (Streamlit)

Após preparar as imagens e/ou treinar o modelo, execute:

```bash
streamlit run app/streamlit_app.py
```