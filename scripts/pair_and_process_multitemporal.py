import os
import argparse
from glob import glob
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from skimage.filters import threshold_otsu
from tqdm import tqdm

# Converte uma imagem PIL para um array numpy RGB.
def to_rgb_array(pil_image):
    return np.array(pil_image.convert("RGB"))

# Calcula a máscara HSV com os limites fornecidos.
def compute_mask_hsv(img_rgb, hmin, smin, vmin, hmax, smax, vmax):
    # img_rgb é RGB; OpenCV usa BGR internamente para convert
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)

# Calcula a máscara ExG usando o método Otsu para limiarização.
def exg_mask_from_rgb(img_rgb):
    R = img_rgb[:,:,0].astype(np.float32)
    G = img_rgb[:,:,1].astype(np.float32)
    B = img_rgb[:,:,2].astype(np.float32)
    exg = 2*G - R - B
    # Otsu espera imagem 2D; funciona com exg
    try:
        thr = threshold_otsu(exg)
    except Exception:
        thr = float(np.mean(exg))
    mask = (exg >= thr).astype(np.uint8) * 255
    return mask, float(thr)

# Calcula o percentual de pixels na máscara, se for vazia retorna 0.
def percent_mask(mask):
    total = mask.size
    if total == 0:
        return 0.0
    return round((np.count_nonzero(mask) / total) * 100, 4)

# Encontra todas as subpastas (áreas) na pasta raiz.
def find_area_folders(root):
    outs = []
    if not os.path.exists(root):
        return outs
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            outs.append(path)
    return outs

# Processa um par de imagens e retorna as métricas.
def process_pair(img_path, hsv_params):
    try:
        pil = Image.open(img_path)
        img_rgb = to_rgb_array(pil)
    except Exception:
        return None
    # Espera hsv_params na ordem: hmin, smin, vmin, hmax, smax, vmax
    hmin, smin, vmin, hmax, smax, vmax = hsv_params
    mask_hsv = compute_mask_hsv(img_rgb, hmin, smin, vmin, hmax, smax, vmax)
    pct_hsv = percent_mask(mask_hsv)
    mask_exg, thr_exg = exg_mask_from_rgb(img_rgb)
    pct_exg = percent_mask(mask_exg)
    return {
        "path": img_path,
        "pct_hsv": pct_hsv,
        "pct_exg": pct_exg,
        "exg_thr": thr_exg
    }

# Função principal para processar todas as áreas e salvar os resultados.
def main(args):
    root = args.root
    y1 = str(args.year1)
    y2 = str(args.year2)
    out_metrics = args.out_metrics
    out_change = args.out_change
    hsv_params = (args.hmin, args.smin, args.vmin, args.hmax, args.smax, args.vmax)

    # Cria diretórios de saída se não existirem
    os.makedirs(os.path.dirname(out_metrics) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_change) or ".", exist_ok=True)

    # Encontra todas as áreas (subpastas) na pasta raiz.
    areas = find_area_folders(root)
    rows = []
    change_rows = []
    skipped_areas = []

    # Processa cada área
    for area in tqdm(areas, desc="Áreas"):
        base = os.path.basename(area)
        f1 = os.path.join(area, f"{y1}.jpg")
        f2 = os.path.join(area, f"{y2}.jpg")

        # fallback para outras extensões (png, jpeg etc.)
        if not os.path.exists(f1):
            f1_alt = glob(os.path.join(area, f"{y1}.*"))
            f1 = f1_alt[0] if f1_alt else ""
        if not os.path.exists(f2):
            f2_alt = glob(os.path.join(area, f"{y2}.*"))
            f2 = f2_alt[0] if f2_alt else ""
        if not (f1 and os.path.exists(f1) and f2 and os.path.exists(f2)):
            skipped_areas.append(base)
            continue

        # Processa o par de imagens (2020 e 2024)
        r1 = process_pair(f1, hsv_params)
        r2 = process_pair(f2, hsv_params)
        if r1 is None or r2 is None:
            skipped_areas.append(base)
            continue

        # métricas por imagem
        rows.append({  
            "area": base,
            "year": y1, # Metricas para 2020.
            "file": os.path.basename(f1),
            "path": f1,
            "pct_hsv": r1["pct_hsv"],
            "pct_exg": r1["pct_exg"],
            "exg_thr": r1["exg_thr"]
        })
        rows.append({  
            "area": base,
            "year": y2, # Métricas para 2024
            "file": os.path.basename(f2),
            "path": f2,
            "pct_hsv": r2["pct_hsv"],
            "pct_exg": r2["pct_exg"],
            "exg_thr": r2["exg_thr"]
        })

        # mudança HSV (2020 -> 2024)
        abs_loss_hsv = round(r1["pct_hsv"] - r2["pct_hsv"], 4)
        rel_loss_hsv = None
        if r1["pct_hsv"] > 0:
            rel_loss_hsv = round((abs_loss_hsv / r1["pct_hsv"]) * 100, 4)

        # mudança ExG
        abs_loss_exg = round(r1["pct_exg"] - r2["pct_exg"], 4)
        rel_loss_exg = None
        if r1["pct_exg"] > 0:
            rel_loss_exg = round((abs_loss_exg / r1["pct_exg"]) * 100, 4)

        # registra mudanças
        change_rows.append({
            "area": base,
            f"pct_hsv_{y1}": r1["pct_hsv"],
            f"pct_hsv_{y2}": r2["pct_hsv"],
            "abs_loss_hsv": abs_loss_hsv,
            "rel_loss_hsv_pct": rel_loss_hsv,
            f"pct_exg_{y1}": r1["pct_exg"],
            f"pct_exg_{y2}": r2["pct_exg"],
            "abs_loss_exg": abs_loss_exg,
            "rel_loss_exg_pct": rel_loss_exg,
            "exg_thr_"+y1: r1["exg_thr"],
            "exg_thr_"+y2: r2["exg_thr"],
        })

    # salvar métricas por imagem
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_metrics, index=False)
        print("Métricas salvas:", out_metrics)
    else:
        print("Nenhuma métrica gerada (verifique a estrutura das pastas).")

    # salvar mudanças por área, se não houver mudanças, avisa.
    if change_rows:
        dfc = pd.DataFrame(change_rows)
        dfc.to_csv(out_change, index=False)
        print("Mudanças salvas:", out_change)
    else:
        print("Nenhuma mudança calculada.")

    if skipped_areas:
        print(f"{len(skipped_areas)} áreas puladas (arquivo faltando ou erro):")
        for a in skipped_areas:
            print(" -", a)

# Parâmetros de linha de comando
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/multitemporal", help="pasta com subpastas por area")
    parser.add_argument("--year1", type=int, default=2020)
    parser.add_argument("--year2", type=int, default=2024)
    parser.add_argument("--out_metrics", default="outputs/multitemporal_metrics.csv")
    parser.add_argument("--out_change", default="outputs/multitemporal_change_2020_2024.csv")
    parser.add_argument("--hmin", type=int, default=15)
    parser.add_argument("--hmax", type=int, default=95)
    parser.add_argument("--smin", type=int, default=21)
    parser.add_argument("--smax", type=int, default=255)
    parser.add_argument("--vmin", type=int, default=20)
    parser.add_argument("--vmax", type=int, default=255)
    args = parser.parse_args()
    main(args)
