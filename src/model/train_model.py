import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report
from tqdm import tqdm

CSV_PATH = "outputs/labels.csv"
MODEL_OUT = "outputs/model_cerrado_resnet18.pth"


class CerradoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.label_map = {"vegetacao": 0, "desmatamento": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        label = self.label_map[row["label"]]

        if self.transform:
            img = self.transform(img)

        return img, label


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42)  # shuffle
    split = int(len(df) * 0.8)
    return df[:split], df[split:]


def main():
    train_df, test_df = load_data(CSV_PATH)

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = CerradoDataset(train_df, transform_train)
    test_ds = CerradoDataset(test_df, transform_test)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔥 Usando dispositivo: {device.upper()}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5
    print("\n🚀 Iniciando treinamento...\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        progress = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}", unit="batch")

        for imgs, labels in progress:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        print(f"✅ Época {epoch+1} concluída | Loss médio: {running_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
            truths.extend(labels.numpy())

    print("\n📊 CLASSIFICATION REPORT:")
    print(classification_report(truths, predictions, target_names=["vegetacao", "desmatamento"]))

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\n💾 Modelo salvo em: {MODEL_OUT}")
    print("🏁 Treinamento finalizado com sucesso!\n")


if __name__ == "__main__":
    main()
