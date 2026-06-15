import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
from glob import glob
import pandas as pd
from skimage.filters import threshold_otsu
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt


# CONFIGURAÇÃO GERAL
st.set_page_config(layout="wide", page_title="SatVision Cerrado — Dashboard Final")
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/overlays", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# Datasets disponíveis
DATASETS = {
    "EuroSAT": "data/processed",
    "Cerrado": "data/multitemporal"
}
dataset_choice = st.sidebar.radio("📂 Escolha o Dataset:", list(DATASETS.keys()))
ROOT_PROCESSED = DATASETS[dataset_choice]

OUT_CSV = "outputs/vegetation_report_final.csv"

# FUNÇÕES AUXILIARES
def to_rgb_array(pil_image):
    return np.array(pil_image.convert("RGB"))

def compute_mask_hsv(img_rgb, hmin, smin, vmin, hmax, smax, vmax):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)

def exg_mask_from_rgb(img_rgb):
    R, G, B = [img_rgb[:, :, i].astype(np.float32) for i in range(3)]
    exg = 2 * G - R - B
    try:
        thr = threshold_otsu(exg)
    except Exception:
        thr = np.mean(exg)
    mask = (exg >= thr).astype(np.uint8) * 255
    return mask, float(thr)

def overlay_image(img_rgb, mask, color=(0, 255, 0), alpha=0.6):
    overlay = img_rgb.copy()
    overlay[mask > 0] = color
    return (img_rgb * (1 - alpha) + overlay * alpha).astype(np.uint8)

def percent_mask(mask):
    total = mask.size
    return round((np.count_nonzero(mask) / total) * 100, 2) if total > 0 else 0.0

# MODELO DE IA
@st.cache_resource
def load_model(path="outputs/model_cerrado_resnet18.pth"):
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m

MODEL = load_model()

def predict_img(img_rgb, model=MODEL):
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    x = t(img_rgb).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1).numpy()[0]
    return int(prob.argmax()), prob

def find_area_folders(root):
    return sorted([
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

# INTERFACE PRINCIPAL
st.title("SatVision Cerrado — Detecção de Vegetação")

st.sidebar.header("Fonte da Imagem")
mode = st.sidebar.radio("Escolha:", ("Upload", "Imagem Processada"))

# Upload manual 
img_rgb, img_name, img_path = None, None, None
if mode == "Upload":
    uploaded = st.sidebar.file_uploader("Carregue uma imagem (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img_rgb = to_rgb_array(Image.open(uploaded))
        img_name = uploaded.name
# Escolha de imagem processada
else:
    classes = ["Selecione"] + sorted(
        [d for d in os.listdir(ROOT_PROCESSED) if os.path.isdir(os.path.join(ROOT_PROCESSED, d))]
    )
    cls = st.sidebar.selectbox("Classe / Área", classes)
    if cls != "Selecione":
        imgs = sorted(glob(os.path.join(ROOT_PROCESSED, cls, "*.jpg")))
        sel = st.sidebar.selectbox("Imagem", imgs)
        if sel:
            img_rgb = to_rgb_array(Image.open(sel))
            img_name = os.path.basename(sel)
            img_path = sel

if img_rgb is None:
    st.info("Nenhuma imagem carregada.")
    st.stop()

# PARÂMETROS HSV
st.sidebar.header("Ajuste HSV (manual)")
hmin = st.sidebar.slider("H min", 0, 179, 15)
hmax = st.sidebar.slider("H max", 0, 179, 95)
smin = st.sidebar.slider("S min", 0, 255, 21)
vmin = st.sidebar.slider("V min", 0, 255, 20)
smax, vmax = 255, 255

method = st.sidebar.radio("Método de Detecção", ["HSV Manual", "Auto ExG (Otsu)"])
st.sidebar.markdown("---")

# PROCESSAMENTO DE IMAGEM
if method == "HSV Manual":
    mask = compute_mask_hsv(img_rgb, hmin, smin, vmin, hmax, smax, vmax)
    overlay = overlay_image(img_rgb, mask, (0, 255, 0))
    pct = percent_mask(mask)
    color = "verde"
else:
    mask, thr = exg_mask_from_rgb(img_rgb)
    overlay = overlay_image(img_rgb, mask, (255, 165, 0))
    pct = percent_mask(mask)
    color = "laranja"

# PREDIÇÃO DA IA
pred, probs = predict_img(img_rgb)
label_pred = "vegetação" if pred == 0 else "desmatamento"
st.markdown(f"**Predição (IA):** {label_pred}")
st.caption(f"Confiança: Vegetação={probs[0]:.3f}, Desmatamento={probs[1]:.3f}")

# 🖼️ EXIBIÇÃO PRINCIPAL
col1, col2 = st.columns(2)
with col1:
    st.subheader("Imagem Original")
    st.image(img_rgb, use_container_width=True)
with col2:
    st.subheader(f"Resultado ({method}) — {pct}% de vegetação")
    st.image(overlay, use_container_width=True)
    st.caption(f"Máscara: {color} = vegetação detectada")

st.markdown("---")

# COMPARAÇÃO MULTITEMPORAL 2020 x 2024
st.header("🕒 Comparação Multitemporal (Cerrado)")

if dataset_choice == "Cerrado":
    areas = find_area_folders("data/multitemporal")
    area = st.selectbox("Selecione a área", areas)
    if area:
        files = sorted([f for f in os.listdir(area) if f.endswith(".jpg") or f.endswith(".png")])
        years = [os.path.splitext(f)[0] for f in files]
        if len(years) >= 2:
            y1 = st.selectbox("Ano 1 (base)", years, index=0)
            y2 = st.selectbox("Ano 2 (comparar)", years, index=len(years)-1)

            p1, p2 = os.path.join(area, f"{y1}.jpg"), os.path.join(area, f"{y2}.jpg")
            img1 = np.array(Image.open(p1).convert("RGB"))
            img2 = np.array(Image.open(p2).convert("RGB"))

            # máscaras (usar os parâmetros HSV atuais)
            mask1 = compute_mask_hsv(img1, hmin, smin, vmin, hmax, smax, vmax) > 0
            mask2 = compute_mask_hsv(img2, hmin, smin, vmin, hmax, smax, vmax) > 0

            # categorias: loss / gain / stable / rest
            loss = (mask1 == True) & (mask2 == False)
            gain = (mask1 == False) & (mask2 == True)
            stable = (mask1 == True) & (mask2 == True)
            rest = (mask1 == False) & (mask2 == False)

            # heatmap RGB (visual)
            h, w = mask1.shape
            heat = np.zeros((h, w, 3), dtype=np.uint8)
            heat[stable] = [0, 200, 0]      # verde = estável
            heat[loss]   = [200, 0, 0]      # vermelho = perda
            heat[gain]   = [0, 0, 200]      # azul = ganho
            heat[rest]   = [200, 255, 200]  # claro = sem vegetação

            # contagens e percentuais
            total_px = h * w
            cnt_loss = int(loss.sum()); pct_loss = round(cnt_loss/total_px*100, 3)
            cnt_gain = int(gain.sum()); pct_gain = round(cnt_gain/total_px*100, 3)
            cnt_stable = int(stable.sum()); pct_stable = round(cnt_stable/total_px*100, 3)
            cnt_rest = int(rest.sum()); pct_rest = round(cnt_rest/total_px*100, 3)

            # Pergunta ao usuário (opcional) resolução para converter em área real
            meters_per_pixel = st.number_input(
                "Resolução espacial (metros/pixel) — informe para calcular área (ex.: 10 para Sentinel-2 10m)",
                min_value=0.0, value=0.0, step=0.1, help="Se deixar 0, só mostra porcentagens (sem área em ha)."
            )

            def px_to_ha(pixels, mpp):
                if mpp <= 0: return None
                return round((pixels * (mpp ** 2)) / 10000.0, 4)  # hectares

            ha_loss = px_to_ha(cnt_loss, meters_per_pixel)
            ha_gain = px_to_ha(cnt_gain, meters_per_pixel)
            ha_stable = px_to_ha(cnt_stable, meters_per_pixel)

            # predições IA para cada imagem
            pred1, probs1 = predict_img(img1)
            pred2, probs2 = predict_img(img2)

            # Exibição lado-a-lado
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                st.subheader(f"{y1}")
                st.image(img1, use_container_width=True)
                st.caption(f"IA: {'vegetação' if pred1==0 else 'desmatamento'} (conf: {probs1[0]:.3f}/{probs1[1]:.3f})")
                st.write("Imagem usada como referência (ano base). Ajuste HSV à esquerda se precisar.")
            with c2:
                st.subheader(f"{y2}")
                st.image(img2, use_container_width=True)
                st.caption(f"IA: {'vegetação' if pred2==0 else 'desmatamento'} (conf: {probs2[0]:.3f}/{probs2[1]:.3f})")
                st.write("Imagem do ano comparado. Ver diferenças no heatmap à direita.")
            with c3:
                st.subheader("Heatmap (diferença)")
                fig, ax = plt.subplots(figsize=(4,4))
                ax.imshow(heat)
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

                # legenda simples (desenho)
                st.markdown("**Legenda (Heatmap):**")
                st.write("- Verde (stable): Vegetação mantida")
                st.write("- Vermelho (loss): Perda de vegetação (desmatamento)")
                st.write("- Azul (gain): Ganho de vegetação (regeneração ou cultivo)")
                st.write("- Claro: Áreas sem vegetação em ambos")

            # resumo numérico
            st.markdown("### Resumo Quantitativo")
            st.write(f"Total pixels: **{total_px}**")
            st.write(f"Perda: **{cnt_loss} px** ({pct_loss}%)" + (f" ≈ {ha_loss} ha" if ha_loss is not None else ""))
            st.write(f"Ganho: **{cnt_gain} px** ({pct_gain}%)" + (f" ≈ {ha_gain} ha" if ha_gain is not None else ""))
            st.write(f"Estável (vegetação mantida): **{cnt_stable} px** ({pct_stable}%)" + (f" ≈ {ha_stable} ha" if ha_stable is not None else ""))
            st.write(f"Sem vegetação (ambos): **{cnt_rest} px** ({pct_rest}%)")

            # delta simples
            abs_loss_pct = round(pct_stable - pct_stable, 3)  # valor simbólico; preferir abs_loss = pct of vegetação entre anos:
            veget_pct_y1 = round((mask1.sum()/total_px)*100,3)
            veget_pct_y2 = round((mask2.sum()/total_px)*100,3)
            abs_change = round(veget_pct_y2 - veget_pct_y1, 3)
            st.markdown(f"**Vegetação {y1}:** {veget_pct_y1}%   —   **Vegetação {y2}:** {veget_pct_y2}%   —   **Δ (y2 - y1):** {abs_change}%")

            # botão salvar heatmap
            if st.button("💾 Salvar Heatmap"):
                out_path = f"outputs/plots/heatmap_{os.path.basename(area)}_{y1}_{y2}.png"
                cv2.imwrite(out_path, cv2.cvtColor(heat, cv2.COLOR_RGB2BGR))
                st.success(f"Heatmap salvo em: {out_path}")

st.caption("© SatVision Cerrado — APS 2025/2 — Universidade Paulista (UNIP)")
