import streamlit as st
import os
from glob import glob
from PIL import Image
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = st.sidebar.radio("Fonte", [
    os.path.join(BASE_DIR, "../data/multitemporal"),
    os.path.join(BASE_DIR, "../data/raw/eurosat")
])
OUTPUT_CSV = os.path.join(BASE_DIR, "../outputs/labels.csv")

st.set_page_config(page_title="SatVision Cerrado — Rotulador", layout="wide")
st.title("🌱 Rotulador de Imagens — SatVision Cerrado")

# Carregar imagens
areas = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
area = st.sidebar.selectbox("Selecione a área", areas)

if area:
    imgs = sorted(glob(os.path.join(DATA_DIR, area, "*.jpg")))
    idx = st.sidebar.number_input("Índice da imagem", 0, len(imgs)-1, 0, 1)
    img_path = imgs[idx]
    img = Image.open(img_path)
    st.image(img, use_container_width=True)

    st.subheader("Selecione a classe:")
    label = st.radio("Classe", ["vegetacao", "desmatamento"])

    if st.button("Salvar rótulo"):
        os.makedirs("outputs", exist_ok=True)
        row = {"path": img_path, "label": label}
        if os.path.exists(OUTPUT_CSV):
            df = pd.read_csv(OUTPUT_CSV)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(OUTPUT_CSV, index=False)
        st.success(f"Rótulo salvo! ({label})")
