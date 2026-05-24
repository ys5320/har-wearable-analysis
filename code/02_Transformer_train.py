import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import math
import os

# ---- paths ----
DATA_DIR = '/rds/general/user/ys5320/home/har_project/data/UCI HAR Dataset'
FIG_DIR  = '/rds/general/user/ys5320/home/har_project/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# ---- device ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ---- load data ----
def load_signals(split):
    sig_dir = os.path.join(DATA_DIR, split, 'Inertial Signals')
    signals = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]
    data = []
    for sig in signals:
        path = os.path.join(sig_dir, f'{sig}_{split}.txt')
        data.append(np.loadtxt(path))
    return np.stack(data, axis=1).astype(np.float32)  # (n, 9, 128)

def load_labels(split):
    path = os.path.join(DATA_DIR, split, f'y_{split}.txt')
    return np.loadtxt(path).astype(np.int64) - 1

print('Loading data...')
X_train = load_signals('train')
y_train = load_labels('train')
X_test  = load_signals('test')
y_test  = load_labels('test')
print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')

# normalise
mean = X_train.mean(axis=(0, 2), keepdims=True)
std  = X_train.std(axis=(0, 2), keepdims=True)
X_train = (X_train - mean) / (std + 1e-8)
X_test  = (X_test  - mean) / (std + 1e-8)

# ---- positional encoding ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # create fixed positional encoding matrix
        # shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices: cos

        # register as buffer — not a parameter, but saved with model
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ---- model ----
class HARClassifierTransformer(nn.Module):
    def __init__(self, n_channels=9, d_model=64, n_heads=4,
                 n_layers=2, d_ff=128, dropout=0.1, n_classes=6):
        super().__init__()

        # project raw 9 channels to d_model dimensions
        # this is the input embedding — like word embedding in NLP
        self.input_proj = nn.Linear(n_channels, d_model)

        # positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=128, dropout=dropout)

        # transformer encoder — stack of n_layers attention blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True    # pre-norm: LayerNorm before attention, more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # classification head
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (batch, 9, 128)
        # Transformer wants (batch, seq_len, features) = (batch, 128, 9)
        x = x.permute(0, 2, 1)

        # project 9 channels → d_model=64
        x = self.input_proj(x)           # (batch, 128, 64)

        # add positional encoding
        x = self.pos_enc(x)              # (batch, 128, 64)

        # transformer encoder — self attention across 128 timepoints
        x = self.transformer(x)          # (batch, 128, 64)

        # aggregate sequence → single vector
        # mean pooling across all timepoints (better than just last timestep)
        x = x.mean(dim=1)               # (batch, 64)

        x = self.norm(x)
        x = self.dropout(x)
        return self.classifier(x)        # (batch, 6)

# ---- training ----
def train_model(model, X_tr, y_tr, n_epochs=100, batch_size=64, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_tr).to(device)
    y_t = torch.tensor(y_tr).to(device)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        losses.append(epoch_loss / len(loader))
        if (epoch+1) % 10 == 0:
            print(f'  Epoch {epoch+1}/{n_epochs}, loss: {losses[-1]:.4f}')
    return losses

# ---- train and evaluate ----
print('\nTraining Transformer...')
model = HARClassifierTransformer(
    n_channels=9, d_model=64, n_heads=4,
    n_layers=2, d_ff=128, dropout=0.1, n_classes=6
)
print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')
losses = train_model(model, X_train, y_train, n_epochs=100)

model.eval()
with torch.no_grad():
    X_test_t = torch.tensor(X_test).to(device)
    preds = model(X_test_t).argmax(dim=1).cpu().numpy()

acc = accuracy_score(y_test, preds)
print(f'\nTransformer test accuracy: {acc:.3f}')
print(f'LSTM test accuracy:        0.893')
print(f'Chance level:              {1/6:.3f}')

# ---- confusion matrix ----
class_names = ['Walking', 'Walk Up', 'Walk Down', 'Sitting', 'Standing', 'Laying']
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_yticklabels(class_names)
for i in range(6):
    for j in range(6):
        ax.text(j, i, cm[i,j], ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'Transformer Confusion Matrix (acc={acc:.3f})')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/transformer_confusion.png', dpi=150)

# ---- loss curve ----
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(losses)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Transformer training loss')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/transformer_loss.png', dpi=150)
print('Figures saved.')