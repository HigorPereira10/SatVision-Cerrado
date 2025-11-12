# 🌎 EcoVision Cerrado — Análise Multitemporal e Detecção de Vegetação

### Projeto APS — Engenharia da Computação / Ciência de Dados  
**Autores:**  
- 🧠 **Higor Pereira** — Pré-processamento, análise multitemporal e interface Streamlit  
- 🤖 **[Colega]** — Modelagem e treinamento da CNN (Inteligência Artificial)

---

## 📘 Descrição Geral

O **EcoVision Cerrado** é um sistema de análise multitemporal que utiliza **imagens de satélite** para **detectar e quantificar mudanças na cobertura vegetal** ao longo do tempo, com foco no bioma **Cerrado brasileiro**.

O projeto combina **processamento de imagens (OpenCV, RasterIO, NumPy)**, **métricas de vegetação (HSV e ExG)** e **análise temporal**, com integração futura a uma **Rede Neural Convolucional (CNN)** para automação da classificação.

---

## 🧱 Estrutura do Projeto

ecovision-cerrado/
│
├── app/
│ └── streamlit_app.py # Interface interativa (Streamlit)
│
├── data/
│ ├── raw/ # Imagens brutas (originais .tif)
│ │ ├── area_001/
│ │ └── area_002/
│ ├── multitemporal/ # Imagens processadas (.jpg)
│ │ ├── area_001/
│ │ └── area_002/
│ ├── EuroSAT/ # Dataset público (fase de calibração)
│ └── EuroSAT_processed/
│
├── outputs/
│ ├── plots/ # Gráficos e heatmaps gerados
│ ├── metrics_multitemporal.csv # Métricas quantitativas
│ ├── multitemporal_change_2020_2024.csv
│ ├── vegetation_report.csv
│ └── ...
│
├── scripts/ # Scripts principais do pipeline
│ ├── batch_process_metrics.py
│ ├── metrics_summary_by_class.py
│ ├── pair_and_process_multitemporal.py
│ ├── plot_multitemporal_diff.py
│ ├── summary_vegetation.py
│ └── tif_to_jpg_512.py
│
├── src/ # Módulos auxiliares
│ ├── preprocess.py
│ ├── hsv_mask.py
│ ├── utils.py
│ ├── extract_eurosat.py
│ └── download_dataset.py
│
├── models/ # CNN (será adicionada pelo colega)
│
├── .gitignore
├── README.md
└── requirements.txt


---

## 🛰️ Etapas Realizadas

### **1️⃣ Fase de Calibração — Dataset EuroSAT**

Para calibrar os índices e limiares de detecção de vegetação, foi utilizado o dataset público **EuroSAT RGB (2750 imagens)**.  
Durante essa etapa foram analisadas diferentes classes (Forest, Pasture, River, Highway, etc.), aplicando técnicas de:

- Conversão HSV (Hue, Saturation, Value)  
- Índice ExG (Excess Green Index)  
- Equalização e limiares automáticos (Otsu)  

Os resultados estão em `outputs/plots/eurosat_analysis/`, incluindo:
- `box_exg_by_class.png`  
- `box_hsv_by_class.png`  
- `scatter_hsv_vs_exg.png`  

Esses gráficos foram usados para determinar **faixas de cor e saturação ideais** para identificar vegetação e não-vegetação.

---

### **2️⃣ Fase de Coleta — Imagens Reais do Cerrado**

Foram utilizadas **imagens do satélite Sentinel-2** obtidas via **Google Earth Engine (GEE)**, correspondentes aos anos **2020** e **2024**, em duas regiões:

| Área | Localização | Resolução | Formato | Fonte |
|------|--------------|------------|----------|--------|
| `area_001` | Próximo a Brasília (GO) | 10m | `.tif` | Sentinel-2 |
| `area_002` | Região agrícola (GO-MT) | 10m | `.tif` | Sentinel-2 |

Cada par de imagens foi baixado via GEE Script e exportado para o Google Drive, depois movido para `data/raw/`.

---

### **3️⃣ Fase de Pré-Processamento**

Os arquivos `.tif` foram convertidos para `.jpg` padronizados de 512×512 pixels com o script:

```bash
python scripts/tif_to_jpg_512.py "data/raw/area_001/area_001_2020.tif" "data/multitemporal/area_001/2020.jpg"

Resultado:

data/multitemporal/area_001/2020.jpg

data/multitemporal/area_001/2024.jpg

data/multitemporal/area_002/2020.jpg

data/multitemporal/area_002/2024.jpg

4️⃣ Fase de Processamento Multitemporal

Com as imagens convertidas, foi realizado o cálculo das métricas de vegetação e a diferença temporal (2020 → 2024):

python scripts/pair_and_process_multitemporal.py --root data/multitemporal --year1 2020 --year2 2024 --out_metrics outputs/multitemporal_metrics.csv


Saídas:

outputs/multitemporal_metrics.csv

outputs/multitemporal_change_2020_2024.csv

Esses arquivos registram a proporção de vegetação detectada em cada área e sua variação percentual.

5️⃣ Fase de Visualização — Heatmaps e Diferenças

Gerado com:
python scripts/plot_multitemporal_diff.py

📊 Resultado:
outputs/plots/heatmap_diff_area_002_2020_2024.png

🟥 Vermelho: perda de vegetação

🟩 Verde: vegetação mantida

🟦 Azul: ganho de vegetação

Essas diferenças são claramente visíveis, representando mudanças reais na cobertura do solo entre os dois anos analisados.

🤖 Integração Futura — CNN (Inteligência Artificial)

A próxima etapa do projeto será conduzida pelo colega responsável pela IA, que implementará e treinará uma CNN (Convolutional Neural Network) para classificar automaticamente as áreas em:

Floresta 🌳

Pastagem 🌾

Agricultura 🚜

Área urbana 🏙️

Água / Rio 💧

O modelo será salvo como:
models/ecovision_model.h5

e integrado ao app Streamlit (app/streamlit_app.py), permitindo análises automáticas e comparações temporais inteligentes.

🌱 Objetivos Alcançados até o Momento
Etapa	Status	Descrição
Estruturação do projeto	✅	Organização modular e limpa
Coleta e processamento de imagens reais	✅	Feito via Google Earth Engine
Calibração HSV / ExG (EuroSAT)	✅	Parâmetros validados
Análise temporal (2020–2024)	✅	Heatmaps e métricas geradas
Preparação para IA	✅	Dados prontos para treinamento

🧩 Tecnologias Utilizadas
Categoria	Ferramentas
Linguagem principal	Python 3.11
Processamento de imagens	OpenCV, Pillow, Rasterio
Análise de dados	NumPy, Pandas, Matplotlib
Visualização e Interface	Streamlit
Geoprocessamento	Google Earth Engine
IA (próxima etapa)	TensorFlow / Keras


🧠 Estrutura Lógica (Pipeline)
graph TD;
    A[Coleta no GEE] --> B[Pré-processamento .tif → .jpg];
    B --> C[Extração de índices (HSV, ExG)];
    C --> D[Geração de métricas e CSVs];
    D --> E[Visualização (heatmap)];
    E --> F[Preparação para CNN];

🗂️ Reprodutibilidade (como executar)
# Criar ambiente virtual
python -m venv .venv
.\.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Converter imagens .tif para .jpg
python scripts/tif_to_jpg_512.py "data/raw/area_001/area_001_2020.tif" "data/multitemporal/area_001/2020.jpg"

# Gerar métricas e comparações
python scripts/pair_and_process_multitemporal.py --root data/multitemporal --year1 2020 --year2 2024 --out_metrics outputs/multitemporal_metrics.csv

# Criar heatmap de diferença
python scripts/plot_multitemporal_diff.py


📈 Resultados Visuais
Ano 2020	Ano 2024	Diferença (2020–2024)

	

