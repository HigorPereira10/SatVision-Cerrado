import os
import argparse
from PIL import Image
from tqdm import tqdm

# Compatibilidade com diferentes versões do Pillow
try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    try:
        RESAMPLE_FILTER = Image.LANCZOS
    except AttributeError:
        RESAMPLE_FILTER = Image.BICUBIC 

# Lista todos os arquivos de imagem em uma pasta e usa yield para economizar memória.
def list_image_files(folder):
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                yield os.path.join(root, f)

# Deduz a classe da imagem a partir do caminho do arquivo.
def get_class_from_path(path, input_root):
    rel = os.path.relpath(path, input_root)
    parts = rel.split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return "unknown"

# Define os parametros de entrada, saída, tamanho alvo, classes e max por classe.
def walk_and_resize(input_dir, output_dir, target_size=(512,512), classes=None, max_per_class=None):
    os.makedirs(output_dir, exist_ok=True)
    counters = {}
    files = list(list_image_files(input_dir))
    pbar = tqdm(files, desc="Arquivos", unit="img")
    for in_path in pbar:
        cls = get_class_from_path(in_path, input_dir)
        if classes and cls not in classes:
            continue
        counters.setdefault(cls, 0)
        if max_per_class and counters[cls] >= max_per_class:
            continue

        # Define o caminho de saída
        rel_dir = os.path.relpath(os.path.dirname(in_path), input_dir)
        out_dir = os.path.join(output_dir, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_name = os.path.splitext(os.path.basename(in_path))[0] + ".jpg"
        out_path = os.path.join(out_dir, out_name)   # Monta o nome final da imagem (mantém o mesmo nome original, só muda a extensão pra .jpg).

        # Redimensiona e salva a imagem como JPEG
        try:
            img = Image.open(in_path).convert("RGB")
            img = img.resize(target_size, RESAMPLE_FILTER)
            img.save(out_path, "JPEG", quality=90)
            counters[cls] += 1
        except Exception as e:
            print(f"Erro {in_path}: {e}") 

    # Resumo final, mostrando quantas imagens de cada classe foram processadas.
    print("Processamento finalizado. Resumo por classe:")
    for k,v in counters.items():
        print(f"  {k}: {v} imagens processadas")

# Execução principal
if __name__ == "__main__":
    # Argumentos de linha de comando
    p = argparse.ArgumentParser(description="Pré-processamento: resize e conversão para JPEG")
    p.add_argument("--input", "-i", default="data/raw/EuroSAT/2750", help="pasta de entrada (ex: data/raw/EuroSAT/2750)")
    p.add_argument("--output", "-o", default="data/processed", help="pasta de saída")
    p.add_argument("--size", "-s", default="512x512", help="tamanho alvo WIDTHxHEIGHT (ex: 512x512)")
    p.add_argument("--classes", "-c", default=None, help="lista de classes separadas por vírgula (ex: AnnualCrop,Bareland) ou None para todas")
    p.add_argument("--max-per-class", "-m", type=int, default=None, help="máximo de imagens por classe (amostragem)")
    args = p.parse_args()

    # Extrai largura e altura do argumento size e processa a lista de classes.
    w,h = (int(x) for x in args.size.split("x"))
    classes = None if not args.classes else [c.strip() for c in args.classes.split(",") if c.strip()]
    # Mostra no terminal os parâmetros escolhidos antes de chamar a função principal.
    print("Input:", args.input)
    print("Output:", args.output)
    print("Tamanho:", (w,h))
    print("Classes (None => todas):", classes)
    print("Max por classe:", args.max_per_class)

    walk_and_resize(args.input, args.output, (w,h), classes=classes, max_per_class=args.max_per_class)
