import os
import tensorflow as tf

# Função para baixar e extrair o dataset EuroSAT
def download_eurosat(dest_folder="data/raw/EuroSAT"):
    os.makedirs(dest_folder, exist_ok=True)
    url = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
    print("Baixando EuroSAT (pode demorar)...")
    path = tf.keras.utils.get_file("EuroSAT.zip", origin=url, extract=True)

    # Move os arquivos extraídos para o diretório de destino
    extracted = os.path.join(os.path.dirname(path), "2750")
    if os.path.exists(extracted):
        import shutil
        shutil.copytree(extracted, dest_folder, dirs_exist_ok=True)
        print(f"Dataset extraído em {dest_folder}")
    else:
        print("Erro: extração não encontrada em", extracted)

# Execução principal
if __name__ == "__main__":
    download_eurosat()
