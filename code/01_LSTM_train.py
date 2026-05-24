import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os

# ---- paths ----
DATA_DIR = '/rds/general/user/ys5320/home/har_project/data/UCI HAR Dataset'
FIG_DIR  = '/rds/general/user/ys5320/home/har_project/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# ---- device ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ---- load raw inertial signals ----
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
        arr = np.loadtxt(path)  # (n_samples, 128)
        data.append(arr)
    # stack to (n_samples, 9, 128)
    return np.stack(data, axis=1).astype(np.float32)

def load_labels(split):
    path = os.path.join(DATA_DIR, split, f'y_{split}.txt')
    return np.loadtxt(path).astype(np.int64) - 1  # 0-indexed

print('Loading data...')
X_train = load_signals('train')  # (7352, 9, 128)
y_train = load_labels('train')
X_test  = load_signals('test')   # (2947, 9, 128)
y_test  = load_labels('test')

print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'X_test:  {X_test.shape},  y_test:  {y_test.shape}')
print(f'Classes: {np.unique(y_train)}')

# normalise per channel using train statistics
mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1, 9, 1)
std  = X_train.std(axis=(0, 2), keepdims=True)
X_train = (X_train - mean) / (std + 1e-8)
X_test  = (X_test  - mean) / (std + 1e-8)

# ---- model ----
class HARClassifierLSTM(nn.Module):
    def __init__(self, n_channels=9, hidden_size=128,
                 num_layers=2, dropout=0.3, n_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        # x: (batch, 9, 128) → LSTM wants (batch, 128, 9)
        x = x.permute(0, 2, 1)
        output, _ = self.lstm(x)
        x = output[:, -1, :]  # take last timepoint
        x = self.dropout(x)
        return self.classifier(x)

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
print('\nTraining LSTM...')
model = HARClassifierLSTM(n_channels=9, hidden_size=128,
                           num_layers=2, dropout=0.3, n_classes=6)
losses = train_model(model, X_train, y_train, n_epochs=100)

model.eval()
with torch.no_grad():
    X_test_t = torch.tensor(X_test).to(device)
    preds = model(X_test_t).argmax(dim=1).cpu().numpy()

acc = accuracy_score(y_test, preds)
print(f'\nLSTM test accuracy: {acc:.3f}')
print(f'Chance level:       {1/6:.3f}')

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
ax.set_title(f'LSTM Confusion Matrix (acc={acc:.3f})')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/lstm_confusion.png', dpi=150)

# ---- loss curve ----
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(losses)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('LSTM training loss')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/lstm_loss.png', dpi=150)
print('Figures saved.')