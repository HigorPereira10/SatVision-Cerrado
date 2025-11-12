# scripts/normalize_labels_paths.py
import pandas as pd
from pathlib import Path
import shutil

BASE = Path.cwd()
csv_orig = BASE / "outputs" / "labels.csv"
csv_backup = BASE / "outputs" / "labels_backup.csv"
csv_fixed = BASE / "outputs" / "labels_fixed.csv"

# pastas onde vamos procurar arquivos pelo nome
SEARCH_DIRS = [
    BASE / "data" / "processed",
    BASE / "data" / "multitemporal",
    BASE / "data" / "raw",
    BASE / "data" / "raw" / "EuroSAT",
]

def find_local_path(filename):
    # procura recursivamente nas pastas SEARCH_DIRS pelo filename (ex: Forest_1006.jpg)
    for d in SEARCH_DIRS:
        if not d.exists():
            continue
        for p in d.rglob(filename):
            # devolve caminho **relativo** ao BASE
            try:
                return str(p.relative_to(BASE))
            except Exception:
                return str(p)
    return None

def normalize():
    if not csv_orig.exists():
        print("Arquivo não encontrado:", csv_orig)
        return

    # backup
    if not csv_backup.exists():
        shutil.copy2(csv_orig, csv_backup)
        print("Backup criado em:", csv_backup)
    else:
        print("Backup já existe:", csv_backup)

    df = pd.read_csv(csv_orig)
    if "path_fixed" in df.columns:
        print("Observação: coluna 'path_fixed' já existe — vamos sobrescrevê-la.")
    paths_fixed = []
    not_found = []

    for idx, row in df.iterrows():
        orig = str(row.get("path", "") or "")
        p = Path(orig)
        # se o caminho já existe (relativo ou absoluto), converte para relativo quando possível
        if p.exists():
            try:
                rel = str(p.relative_to(BASE))
            except Exception:
                rel = str(p)
            paths_fixed.append(rel)
            continue

        # se não existe, tenta procurar pelo nome do arquivo no projeto
        fname = p.name
        candidate = find_local_path(fname)
        if candidate:
            paths_fixed.append(candidate)
        else:
            paths_fixed.append(orig)   # mantém original para inspeção manual
            not_found.append(orig)

    df["path_fixed"] = paths_fixed
    df.to_csv(csv_fixed, index=False)
    print(f"Arquivo com caminhos normalizados salvo em: {csv_fixed}")
    print(f"Total linhas: {len(df)} | Não encontrados: {len(not_found)}")
    if len(not_found) > 0:
        print("Exemplos não encontrados (até 10):")
        for v in not_found[:10]:
            print(" ", v)

    # Se ninguém ficou faltando, opcionalmente sobrescrever
    if len(not_found) == 0:
        # atualizar coluna 'path' para ser a 'path_fixed' e salvar de volta
        df["path"] = df["path_fixed"]
        df.drop(columns=["path_fixed"], inplace=True)
        df.to_csv(csv_orig, index=False)
        print("Todos os caminhos atualizados e sobrescrevendo outputs/labels.csv")
    else:
        print("Como houve caminhos não encontrados, NÃO foi sobrescrito outputs/labels.csv.")
        print("Revise outputs/labels_fixed.csv e corrija manualmente os casos listados.")

if __name__ == "__main__":
    normalize()
