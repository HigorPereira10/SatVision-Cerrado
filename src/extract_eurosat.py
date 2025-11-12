import zipfile
import os
import sys

# Caminho do arquivo zip do EuroSAT e diretório de saída
zip_path = r"data\\raw\\EuroSAT.zip"  
out_dir = r"data\\raw\\EuroSAT"

# Verifica se o arquivo zip existe
if not os.path.exists(zip_path):
    print("ERRO: arquivo não encontrado em:", zip_path)
    sys.exit(1)

# Extrai o conteúdo do arquivo zip e salva no diretório de saída
os.makedirs(out_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(out_dir)
print("Extraído em:", out_dir)
