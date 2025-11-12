import sys
import os
import numpy as np
import rasterio
from PIL import Image

def normalize_array(arr):

    # Converte o array para foat e substitui valores infinitos ou invalidos por NaN.
    arr = arr.astype(float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    # Determina os percentis 2 e 98 para normalização (valores extremos são ignorados como nuvem ou sombra)
    lo = np.nanpercentile(arr, 2)
    hi = np.nanpercentile(arr, 98)
    # Se não houver variação, usa min e max reais e se ainda não houver, retorna array de zeros(preto).
    if np.isnan(lo) or np.isnan(hi) or hi - lo == 0:
        lo = np.nanmin(arr)
        hi = np.nanmax(arr)
        if np.isnan(lo) or np.isnan(hi) or hi - lo == 0:
            return np.zeros_like(arr, dtype=np.uint8)
    # Normaliza para 0-255, a imagem fia com contraste equilibrado.
    clipped = np.clip(arr, lo, hi)
    norm = (clipped - lo) / (hi - lo + 1e-8)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    return (norm * 255).astype(np.uint8)

# Converte um arquivo TIFF para JPEG redimensionado para 512x512 pixels.
def convert_tif_to_jpg(input_path, output_path, size=(512, 512)):

    # Verifica se o arquivo de entrada existe e cria o diretório de saída se necessário.
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Lê o arquivo TIFF usando rasterio e processa as bandas.
    with rasterio.open(input_path) as src:
        count = src.count
        # se houver menos de 3 bandas, duplica a primeira banda para criar uma imagem RGB, caso contrário, usa as três primeiras bandas.
        if count < 3:
            print(f"Aviso: o arquivo {input_path} tem apenas {count} bandas. Serão duplicadas para RGB.")
            data = np.repeat(src.read(1)[None, :, :], 3, axis=0)
        else:
            data = src.read([1, 2, 3])  # RGB
        # Normaliza cada banda e empilha para formar uma imagem RGB.
        rgb = np.stack([normalize_array(band) for band in data], axis=-1)
        # Salva a imagem como JPEG redimensionada.
        img = Image.fromarray(rgb)
        img = img.resize(size, Image.LANCZOS)
        img.save(output_path, "JPEG", quality=95)
        print(f"✅ Convertido: {output_path}")

# Permite a execução no terinal com argumentos de entrada (.tif) e saída (.jpg).
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python scripts/tif_to_jpg_512.py <entrada.tif> <saida.jpg>")
        sys.exit(1)
    convert_tif_to_jpg(sys.argv[1], sys.argv[2])
