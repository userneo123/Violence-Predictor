# ============================================================
# EARLY VIOLENCE PREDICTION (CNN + LSTM)
# ============================================================


import os
import ast
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

# ------------------ CONFIG ------------------
ROOT = "data/"  

VIOLENCE_DIR    = os.path.join(ROOT, "violenceFrame")
NONVIOLENCE_DIR = os.path.join(ROOT, "nonViolenceFrame")

CSV_V  = os.path.join(VIOLENCE_DIR, "dataset_kaggle.csv")
CSV_NV = os.path.join(NONVIOLENCE_DIR, "dataset_kaggle.csv")

SAVE_PATH = "best_model_arnav_new.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SIZE = 16
BATCH_SIZE  = 16
EPOCHS      = 25
LR          = 1e-4

# ------------------ TRANSFORMS ------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------ DATASET ------------------
class ViolenceDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        source = row["source"]
        label  = float(row["label"])

        base_dir = VIOLENCE_DIR if source == "violence" else NONVIOLENCE_DIR

        raw = str(row["frame_paths"])
        if "|" in raw:
            paths = raw.split("|")
        else:
            try:
                paths = ast.literal_eval(raw)
            except:
                paths = raw.split(",")

        frames = []
        for p in paths:
            full_path = os.path.join(base_dir, p.strip())
            img = Image.open(full_path).convert("RGB")
            img = self.transform(img)
            frames.append(img)

        return torch.stack(frames), torch.tensor(label, dtype=torch.float32)

# ------------------ LOAD DATA ------------------
df_v  = pd.read_csv(CSV_V);  df_v["source"]  = "violence"
df_nv = pd.read_csv(CSV_NV); df_nv["source"] = "nonviolence"

df_v["video_id"]  = "v_"  + df_v["video_id"].astype(str)
df_nv["video_id"] = "nv_" + df_nv["video_id"].astype(str)

df = pd.concat([df_v, df_nv], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=df["video_id"]))

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df   = df.iloc[val_idx].reset_index(drop=True)

train_dataset = ViolenceDataset(train_df, transform_train)
val_dataset   = ViolenceDataset(val_df, transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------ MODEL ------------------
class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        self.lstm    = nn.LSTM(512, 128, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(128, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        features = self.cnn(x).view(B, T, 512)
        out, _ = self.lstm(features)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(1)

# ------------------ SETUP ------------------
model = CNN_LSTM().to(DEVICE)

# freeze CNN
for param in model.cnn.parameters():
    param.requires_grad = False

# class weights
n_neg = (train_df["label"] == 0).sum()
n_pos = (train_df["label"] == 1).sum()
pos_weight = torch.tensor([n_neg / n_pos]).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# train only LSTM + FC
trainable = list(model.lstm.parameters()) + list(model.fc.parameters())
optimizer = torch.optim.Adam(trainable, lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5
)

# ------------------ TRAIN ------------------
best_val_acc = 0

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for frames, labels in train_loader:
        frames, labels = frames.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for frames, labels in val_loader:
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)

            logits = model(frames)
            preds = (torch.sigmoid(logits) > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy().astype(int))
            all_labels.extend(labels.cpu().numpy().astype(int))

    val_acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds)

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | F1: {f1:.4f}")

print("Training Complete")
print(f"Best Model Saved: {SAVE_PATH}")
